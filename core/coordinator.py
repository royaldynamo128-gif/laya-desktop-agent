"""
Pipeline Coordinator for Laya Desktop Agent.
Orchestrates: Audio -> VAD -> STT -> Semantic Parse (+Laya hint) -> Binder
-> Executor -> Observer -> Verifier -> Recovery -> TTS.

Latency architecture:
  - The deterministic SemanticParser is the primary intent engine (~0ms).
  - The Laya neural router runs only for unknown intents / safety
    classification, guarded by a NON-BLOCKING timeout (a hung inference can
    never stall the pipeline — the classic ThreadPoolExecutor.__exit__
    deadlock is structurally impossible here).
  - Observations are TTL-cached; each command performs at most one fresh
    observation instead of three.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging
import threading
import time
import uuid

from core.config import AgentConfig
from core.state import StateManager, AgentState
from audio.vad import VoiceActivityDetector
from audio.capture import AudioCapture
from audio.player import AudioPlayer
from models.manager import ModelManager
from models.laya_engine import LayaDecision
from execution.parameter_binder import ParameterBinder, BoundAction
from execution.executor import ActionExecutor
from verification.observer import DesktopObserver, DesktopObservation
from verification.verifier import ActionVerifier, VerificationResult
from verification.recovery import RecoveryManager
from monitoring.latency_tracker import LatencyTracker, CommandMetrics
from monitoring.cache_reaper import CacheReaper
from monitoring.resource_monitor import ResourceMonitor
from core.status_ipc import StatusBroadcaster

from core.lifecycle import CommandLifecycle, CommandStage, LifecycleManager
from planning.goal_executor import GoalExecutor, GoalResult
from planning.goal_planner import GoalPlanner
from planning.method_registry import MethodRegistry
from planning.speech_corrector import SpeechCorrector
from planning.llm_backend import AutoPlannerBackend, RuleBasedPlannerBackend
from planning.semantic_parser import get_parser
from audio.validator import AudioValidator

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    command_id: str
    transcription: str
    intent: str
    action_executed: Optional[str]
    success: bool
    verified: bool
    response_text: str
    metrics: CommandMetrics
    raw_transcription: str = ""
    normalized_transcription: str = ""


class PipelineCoordinator:
    def __init__(
        self,
        config: AgentConfig,
        state_mgr: StateManager,
        model_mgr: ModelManager,
        vad: VoiceActivityDetector,
        player: AudioPlayer,
        binder: ParameterBinder,
        executor: ActionExecutor,
        observer: DesktopObserver,
        verifier: ActionVerifier,
        recovery_mgr: RecoveryManager,
        tracker: LatencyTracker,
        reaper: CacheReaper,
        monitor: ResourceMonitor,
        status_broadcaster: Optional[StatusBroadcaster] = None,
        diagnostic_mode: bool = False,
        lifecycle_mgr: Optional[LifecycleManager] = None,
        goal_verifier=None,
        goal_executor: Optional[GoalExecutor] = None,
    ):
        self.config = config
        self.state_mgr = state_mgr
        self.model_mgr = model_mgr
        self.vad = vad
        self.player = player
        self.binder = binder
        self.executor = executor
        self.observer = observer
        self.verifier = verifier
        self.recovery_mgr = recovery_mgr
        self.tracker = tracker
        self.reaper = reaper
        self.monitor = monitor
        self.status_broadcaster = status_broadcaster
        self.diagnostic_mode = diagnostic_mode
        self.lifecycle_mgr = lifecycle_mgr
        self.emergency_stop = getattr(executor, "emergency_stop", None)

        # Autonomous Planning & Speech Subsystems
        self.audio_validator = AudioValidator(sample_rate=config.input.sample_rate)
        catalog = getattr(self.binder, "catalog", None)
        self.speech_corrector = SpeechCorrector(catalog=catalog)
        self.parser = get_parser()
        self.planner = GoalPlanner(AutoPlannerBackend())
        switch_threshold = (
            self.config.planner.method_switch_threshold
            if hasattr(self.config, "planner")
            else 2
        )
        goal_cfg = getattr(self.config, "goal", None)
        max_duration = getattr(goal_cfg, "max_total_duration_sec", 120.0) if goal_cfg else 120.0
        if self.goal_executor_available(goal_executor):
            self.goal_executor = goal_executor
        else:
            self.goal_executor = GoalExecutor(
                planner=self.planner,
                coordinator=self,
                observer=self.observer,
                verifier=self.verifier,
                method_registry=MethodRegistry(),
                goal_verifier=goal_verifier,
                method_switch_threshold=switch_threshold,
                max_total_duration_sec=max_duration,
            )

    def goal_executor_available(self, ge) -> bool:
        return ge is not None

    # ------------------------------------------------------------------
    # Compound goal detection (parser-driven, no regex splitting)
    # ------------------------------------------------------------------

    def is_compound_goal(self, text: str) -> bool:
        """True when the utterance decomposes into multiple clauses."""
        if not text:
            return False
        goal_cfg = getattr(self.config, "goal", None)
        if goal_cfg and not getattr(goal_cfg, "autonomous_mode", True):
            return False
        try:
            parsed = self.parser.parse(text)
            return parsed.is_compound
        except Exception as e:
            logger.debug("compound goal parse failed: %s", e)
            return False

    def process_goal(self, goal: str, lifecycle: Optional[CommandLifecycle] = None) -> GoalResult:
        """Direct entry point for executing autonomous goals."""
        return self.goal_executor.execute_goal(goal, lifecycle=lifecycle)

    # ------------------------------------------------------------------
    # Audio pipeline
    # ------------------------------------------------------------------

    def process_audio(self, raw_pcm: bytes, lifecycle: Optional[CommandLifecycle] = None) -> Optional[PipelineResult]:
        """Process recorded microphone audio through the full pipeline."""
        command_id = lifecycle.command_id if lifecycle else str(uuid.uuid4())[:8]
        metrics = CommandMetrics(command_id=command_id)
        snapshot = self.monitor.get_snapshot()
        metrics.vram_used_mb = snapshot.vram_used_mb
        metrics.ram_used_mb = snapshot.process_rss_mb

        self.state_mgr.set_state(AgentState.PROCESSING, reason="Audio received")

        # 1. Signal quality validation & preprocessing
        val_res = self.audio_validator.validate_and_preprocess(raw_pcm)
        metrics.audio_quality_conf = 1.0 if val_res.is_usable else 0.0
        logger.info(
            "[%s] Signal Validation: usable=%s, RMS=%.1f dBFS, Peak=%.1f dBFS, Reason='%s'",
            command_id, val_res.is_usable, val_res.rms_dbfs, val_res.peak_dbfs, val_res.failure_reason,
        )

        if not val_res.is_usable:
            if lifecycle:
                lifecycle.fail(reason=val_res.failure_reason)
            self.state_mgr.set_state(AgentState.IDLE, reason=val_res.failure_reason)
            if self.status_broadcaster:
                self.status_broadcaster.broadcast(
                    state="FAILURE", title="● No audio detected", detail=val_res.failure_reason,
                    next_action="Hold Meta + Alt and speak clearly", level="warning",
                )
            self._speak_response("I didn't hear anything.", metrics)
            return None

        # 2. VAD
        with self.tracker.measure("vad", metrics):
            vad_res = self.vad.process(val_res.preprocessed_pcm)

        if lifecycle:
            lifecycle.transition(CommandStage.VAD_COMPLETE, detail=f"has_speech={vad_res.has_speech}")

        if not vad_res.has_speech:
            logger.info("[%s] No speech detected by VAD. Ignoring.", command_id)
            if lifecycle:
                lifecycle.fail(reason="No speech detected")
            self.state_mgr.set_state(AgentState.IDLE, reason="No speech")
            if self.status_broadcaster:
                self.status_broadcaster.broadcast(
                    state="FAILURE", title="● No speech detected", detail="Audio input had no voiced speech frames",
                    next_action="Hold Meta + Alt and speak clearly", level="warning",
                )
            self._speak_response("No speech detected.", metrics)
            return None

        speech_pcm = vad_res.speech_pcm

        # 3. STT
        if lifecycle:
            lifecycle.transition(CommandStage.STT_START)
        if self.status_broadcaster:
            self.status_broadcaster.broadcast(
                state="TRANSCRIBING", title="◌ Transcribing...", detail="Converting speech to text...", next_action="",
            )

        # Dynamic vocabulary biasing from the app catalog
        whisper_prompt = None
        whisper_hotwords = None
        if hasattr(self.binder, "catalog") and self.binder.catalog:
            try:
                whisper_prompt = self.binder.catalog.generate_whisper_prompt()
                whisper_hotwords = self.binder.catalog.generate_hotwords()
            except Exception as e:
                logger.debug("[%s] Could not generate whisper vocabulary prompt: %s", command_id, e)

        beam = getattr(getattr(self.config, "stt", None), "beam_size", 1)
        with self.tracker.measure("stt", metrics):
            trans_res = self.model_mgr.whisper.transcribe(
                speech_pcm,
                initial_prompt=whisper_prompt,
                hotwords=whisper_hotwords,
                beam_size=beam,
            )
            transcription = trans_res.text if hasattr(trans_res, "text") else str(trans_res[0])
            stt_conf = getattr(trans_res, "confidence", 0.95)

        if lifecycle:
            lifecycle.transition(CommandStage.STT_COMPLETE, detail=f"conf={stt_conf:.2f}")

        raw_tx = getattr(trans_res, "raw_text", transcription)
        metrics.raw_transcription = raw_tx
        metrics.stt_confidence = stt_conf

        if not transcription.strip():
            logger.info("[%s] Empty transcription.", command_id)
            if lifecycle:
                lifecycle.fail(reason="Empty transcription")
            self.state_mgr.set_state(AgentState.IDLE, reason="Empty transcription")
            self._speak_response("I couldn't understand that.", metrics)
            return None

        # Hallucination filter (Whisper artifacts like "Thank you." on silence)
        try:
            from models.whisper_engine import is_hallucination
            if is_hallucination(transcription):
                logger.info("[%s] STT hallucination filtered: '%s'", command_id, transcription)
                if lifecycle:
                    lifecycle.fail(reason="STT hallucination")
                self.state_mgr.set_state(AgentState.IDLE, reason="Hallucination")
                self._speak_response("I couldn't understand that.", metrics)
                return None
        except Exception:
            pass

        # 4. Context-aware speech correction
        raw_tx, normalized_tx = self.speech_corrector.normalize(transcription)
        if normalized_tx != transcription:
            logger.info("[%s] Transcript normalized: '%s' -> '%s'", command_id, transcription, normalized_tx)

        metrics.raw_transcription = raw_tx
        metrics.transcription = normalized_tx
        logger.info("[%s] Transcribed: '%s' -> '%s' (conf: %.2f)", command_id, raw_tx, normalized_tx, stt_conf)

        if lifecycle:
            lifecycle.raw_transcription = raw_tx
            lifecycle.normalized_transcription = normalized_tx
            lifecycle.transcription = normalized_tx
            lifecycle.transition(CommandStage.TRANSCRIPT_READY, detail=normalized_tx)

        if self.status_broadcaster:
            self.status_broadcaster.broadcast(
                state="UNDERSTANDING", title=f"Heard: \"{normalized_tx}\"",
                detail="Resolving desktop command...", next_action="", command_text=normalized_tx,
            )

        return self.process_text_command(normalized_tx, metrics=metrics, command_id=command_id, lifecycle=lifecycle)

    # ------------------------------------------------------------------
    # Text command pipeline
    # ------------------------------------------------------------------

    def process_text_command(
        self,
        command_text: str,
        metrics: Optional[CommandMetrics] = None,
        command_id: Optional[str] = None,
        lifecycle: Optional[CommandLifecycle] = None,
        allow_autonomous_decomposition: bool = True,
    ) -> PipelineResult:
        """Process a text command with full stage tracking."""
        cid = (lifecycle.command_id if lifecycle else None) or command_id or str(uuid.uuid4())[:8]
        m = metrics or CommandMetrics(command_id=cid, transcription=command_text)

        if lifecycle is None:
            if self.lifecycle_mgr:
                lifecycle = self.lifecycle_mgr.new_command()
                cid = lifecycle.command_id
            else:
                lifecycle = CommandLifecycle(command_id=cid)

        # Watchdog: adaptive — goal executor pings on step progress
        def _on_cmd_timeout(lc: CommandLifecycle):
            logger.error("[%s] Command watchdog expired! Stuck at: %s", cid, lc.current_stage.value)
            lc.request_cancel(reason="watchdog timeout")
            self.state_mgr.set_state(AgentState.IDLE, reason="Command watchdog timeout")
            if self.status_broadcaster:
                self.status_broadcaster.broadcast(
                    state="FAILURE", title="✕ Timed Out", detail=f"Stuck at: {lc.current_stage.value}",
                    next_action="Hold Meta + Alt to try again", level="error", command_text=command_text,
                )

        try:
            # --- Compound goal routing -----------------------------------
            if allow_autonomous_decomposition and self.is_compound_goal(command_text):
                logger.info("[%s] Compound goal: '%s' → GoalExecutor", cid, command_text)
                lifecycle.transition(CommandStage.PLANNING, detail=f"Goal: {command_text}")

                # Compound goals can take longer: extend watchdog budget
                if not lifecycle.is_finalized:
                    lifecycle.start_watchdog(timeout_sec=45.0, on_timeout=_on_cmd_timeout)

                goal_start = time.perf_counter()
                with self.tracker.measure("goal_execution", m):
                    goal_res = self.goal_executor.execute_goal(command_text, lifecycle=lifecycle)
                logger.info(
                    "[%s] Compound goal finished in %.1f ms: %s",
                    cid, (time.perf_counter() - goal_start) * 1000.0, goal_res.message,
                )
                m.intent = "autonomous_goal"
                m.target = command_text[:60]

                # Speak the final goal outcome (was silently missing before)
                self._speak_response(goal_res.message, m)
                return self._finalize_result(
                    cid=cid,
                    transcription=command_text,
                    intent="autonomous_goal",
                    tool="goal_executor",
                    success=goal_res.success,
                    verified=goal_res.verified,
                    response_text=goal_res.message,
                    metrics=m,
                    lifecycle=lifecycle,
                )

            # --- Single command path --------------------------------------
            if not lifecycle.is_finalized:
                lifecycle.start_watchdog(timeout_sec=20.0, on_timeout=_on_cmd_timeout)

            lifecycle.transition(CommandStage.LAYA_START)
            self.state_mgr.set_state(AgentState.PROCESSING, reason="Routing")
            if self.status_broadcaster:
                self.status_broadcaster.broadcast(
                    state="UNDERSTANDING", title="◌ Understanding...", detail=f"'{command_text}'",
                    next_action="", command_text=command_text,
                )

            # Routing: semantic parser first (deterministic, ~0ms),
            # neural router only for unknown intents (non-blocking timeout).
            with self.tracker.measure("laya_routing", m):
                decision = self._route(command_text)

            m.intent = decision.intent
            lifecycle.intent = decision.intent
            lifecycle.transition(
                CommandStage.LAYA_COMPLETE,
                detail=f"intent={decision.intent} (conf: {decision.confidence:.2f})",
            )
            logger.info("[%s] Routed intent: %s (conf: %.2f)", cid, decision.intent, decision.confidence)

            # --- Binding ---------------------------------------------------
            bound_action = self.binder.bind(command_text, decision)
            if not bound_action:
                response_text = f"I couldn't find an application or action matching '{command_text}'."
                logger.info("[%s] %s", cid, response_text)
                self._speak_response(response_text, m)
                lifecycle.fail(reason=response_text)
                return self._finalize_result(cid, command_text, decision.intent, None, False, False, response_text, m, lifecycle=lifecycle)

            m.selected_tool = bound_action.tool_name
            m.target = bound_action.target or bound_action.tool_name
            m.entity_resolution_conf = 0.95
            m.intent_confidence = decision.confidence

            verb = {
                "open_application": "Opening",
                "close_application": "Closing",
                "switch_window": "Switching to",
                "file_move": "Moving",
                "file_rename": "Renaming",
                "file_open": "Opening",
                "browser_navigate": "Navigating",
                "browser_search": "Searching",
                "click_element": "Clicking",
            }.get(bound_action.tool_name, "Executing")

            lifecycle.target = m.target
            lifecycle.transition(CommandStage.ENTITY_READY, detail=f"{verb} {m.target}")

            # Clarification requests
            if bound_action.tool_name == "clarify":
                prompt = bound_action.arguments.get("prompt", f"Could you clarify your request: '{command_text}'?")
                logger.info("[%s] Clarification: %s", cid, prompt)
                self._speak_response(prompt, m)
                lifecycle.transition(CommandStage.PLAN_READY, detail="clarify")
                if self.status_broadcaster:
                    self.status_broadcaster.broadcast(
                        state="CLARIFY", title="? Clarification Needed", detail=prompt,
                        next_action="Hold Meta + Alt to answer", level="warning", command_text=command_text,
                    )
                return self._finalize_result(cid, command_text, decision.intent, "clarify", True, True, prompt, m, lifecycle=lifecycle)

            # Diagnostic mode: zero execution
            if self.diagnostic_mode:
                response_text = f"[DIAGNOSTIC] Intent: {decision.intent} | Tool: {bound_action.tool_name} | Target: {m.target}"
                logger.info("[%s] Diagnostic mode: not executing.", cid)
                lifecycle.succeed(detail=response_text)
                return self._finalize_result(cid, command_text, decision.intent, bound_action.tool_name, True, True, response_text, m, lifecycle=lifecycle)

            # Safety confirmation gate
            if decision.requires_confirmation:
                response_text = f"Action '{bound_action.tool_name}' requires confirmation. Execution blocked for safety."
                logger.warning("[%s] %s", cid, response_text)
                self._speak_response(response_text, m)
                lifecycle.fail(reason=response_text)
                return self._finalize_result(cid, command_text, decision.intent, bound_action.tool_name, False, False, response_text, m, lifecycle=lifecycle)

            logger.info(
                "[%s] PIPELINE: HEARD='%s' -> TARGET='%s' (tool=%s)",
                cid, command_text, m.target, bound_action.tool_name,
            )

            # Single cached pre-action observation (was 3 full observations)
            pre_obs = self.observer.observe(delay_sec=0.0)

            # --- Execution --------------------------------------------------
            lifecycle.transition(CommandStage.EXECUTION_START, detail=f"{verb} {m.target}")
            self.state_mgr.set_state(AgentState.EXECUTING, reason=bound_action.tool_name)
            if self.status_broadcaster:
                self.status_broadcaster.broadcast(
                    state="EXECUTING", title=f"{verb}: {m.target}", detail=f"Heard: \"{command_text}\"",
                    next_action="", command_text=command_text,
                )

            with self.tracker.measure("execution", m):
                tool_res = self.executor.execute_action(bound_action)

            lifecycle.transition(CommandStage.EXECUTION_COMPLETE, detail=f"success={tool_res.success}")

            # --- Verification -------------------------------------------------
            lifecycle.transition(CommandStage.VERIFICATION_START)
            self.state_mgr.set_state(AgentState.VERIFYING, reason="Observing state")
            if self.status_broadcaster:
                self.status_broadcaster.broadcast(
                    state="VERIFYING", title="◌ Verifying...", detail=f"Checking {m.target}...",
                    next_action="", command_text=command_text,
                )

            # Any action invalidates the observation cache
            self.observer.invalidate()
            with self.tracker.measure("verification", m):
                obs_delay = 0.05 if bound_action.tool_name in ("open_application", "switch_window", "close_application") else 0.0
                post_obs = self.observer.observe(delay_sec=obs_delay)
                verify_res = self.verifier.verify(bound_action, tool_res, pre_obs, post_obs)

            # --- Recovery ---------------------------------------------------
            if not verify_res.verified:
                with self.tracker.measure("recovery", m):
                    recovery_res, final_verify = self.recovery_mgr.attempt_recovery(
                        action=bound_action,
                        verification=verify_res,
                        execute_fn=lambda a: self.executor.execute_action(a),
                        verify_fn=lambda a, r: self.verifier.verify(a, r, pre_obs, self.observer.observe(delay_sec=0.05)),
                    )
                    if recovery_res:
                        tool_res = recovery_res
                    verify_res = final_verify

            lifecycle.transition(CommandStage.VERIFICATION_COMPLETE, detail=f"verified={verify_res.verified}")

            # --- TTS feedback ------------------------------------------------
            response_text = tool_res.message
            self._speak_response(response_text, m)

            return self._finalize_result(
                cid, command_text, decision.intent, bound_action.tool_name,
                tool_res.success, verify_res.verified, response_text, m,
                lifecycle=lifecycle,
            )

        except Exception as e:
            logger.exception("[%s] Unhandled exception in process_text_command: %s", cid, e)
            if lifecycle:
                lifecycle.fail(reason=str(e))
            if self.status_broadcaster:
                self.status_broadcaster.broadcast(
                    state="FAILURE", title="✕ Error", detail=f"{str(e)[:40]}",
                    next_action="Hold Meta + Alt to try again", level="error", command_text=command_text,
                )
            return self._finalize_result(
                cid=cid, transcription=command_text, intent="error", tool=None,
                success=False, verified=False, response_text=f"An error occurred: {e}",
                metrics=m, lifecycle=lifecycle,
            )
        finally:
            lifecycle.stop_watchdog()
            self.state_mgr.set_state(AgentState.IDLE, reason="Command processing ended")

    # ------------------------------------------------------------------
    # Routing: parser first, neural router as fallback (non-blocking)
    # ------------------------------------------------------------------

    def _route(self, command_text: str) -> LayaDecision:
        """
        Route a command to an intent decision.
        1. SemanticParser (deterministic, 0ms) — primary.
        2. Laya neural router — only for unknown intents, with a strictly
           non-blocking timeout. A hung model can never hang the pipeline:
           the worker is a daemon thread and the result future is abandoned
           (never joined) on timeout.
        """
        try:
            parsed = self.parser.parse(command_text)
            intent = parsed.primary_intent if parsed.clauses else "other"
            confidence = parsed.clauses[0].confidence if parsed.clauses else 0.3
        except Exception as e:
            logger.warning("Parser failed: %s", e)
            intent, confidence = "other", 0.3

        if intent not in ("unknown", "other"):
            return LayaDecision(
                intent=intent,
                confidence=confidence,
                safety_level="safe",
                requires_confirmation=False,
                requires_observation=True,
                raw_decision={"router": "semantic_parser"},
            )

        # Unknown intent → neural router with non-blocking timeout
        laya_engine = getattr(self.model_mgr, "laya", None)
        if laya_engine is not None:
            try:
                if not laya_engine.is_loaded and not getattr(laya_engine, "fallback_only", False):
                    laya_engine.load()

                if laya_engine.is_loaded:
                    result_box: Dict[str, Any] = {}
                    done = threading.Event()

                    def _infer():
                        try:
                            result_box["decision"] = laya_engine.decide(command_text)
                        except Exception as e:
                            result_box["error"] = e
                        finally:
                            done.set()

                    worker = threading.Thread(target=_infer, daemon=True, name="LayaRouter")
                    worker.start()
                    # Abandon-on-timeout: never join a stuck inference
                    if done.wait(timeout=2.0):
                        if "decision" in result_box:
                            return result_box["decision"]
                        logger.warning("Laya router error: %s", result_box.get("error"))
                else:
                    logger.info("Laya router unavailable; using heuristic fallback routing.")
            except Exception as e:
                logger.warning("Laya routing setup failed (%s); heuristic fallback.", e)

        return laya_engine._heuristic_fallback(command_text, 0.0) if laya_engine else self._static_fallback(command_text)

    @staticmethod
    def _static_fallback(command_text: str) -> LayaDecision:
        """Deterministic last-resort router (no model dependency)."""
        lower = command_text.lower()
        if any(w in lower for w in ("open", "launch", "start")):
            intent = "open_application"
        elif any(w in lower for w in ("close", "quit", "exit")):
            intent = "close_application"
        elif any(w in lower for w in ("switch to", "focus", "bring up")):
            intent = "switch_window"
        elif any(w in lower for w in ("search", "youtube", "google")):
            intent = "browser_action"
        elif "screenshot" in lower:
            intent = "screen_query"
        elif any(w in lower for w in ("type", "write", "press")):
            intent = "input_action"
        elif any(w in lower for w in ("volume", "mute")):
            intent = "system_setting"
        else:
            intent = "other"
        return LayaDecision(
            intent=intent, confidence=0.85, safety_level="safe",
            requires_confirmation=False, requires_observation=True,
            raw_decision={"fallback": True}, latency_ms=0.0,
        )

    # ------------------------------------------------------------------
    # Speech helpers
    # ------------------------------------------------------------------

    def _speak_response(self, text: str, metrics: CommandMetrics, return_to_idle: bool = True) -> None:
        """Synthesize and play audio response via Piper on CPU."""
        if not self.config.tts.enabled or not text:
            if return_to_idle:
                self.state_mgr.set_state(AgentState.IDLE, reason="TTS skipped")
            return

        self.state_mgr.set_state(AgentState.SPEAKING, reason="Playing audio response")
        try:
            with self.tracker.measure("tts", metrics):
                wav_path = self.model_mgr.piper.synthesize_to_wav(text)
            if wav_path and wav_path.exists():
                self.player.play_wav(wav_path, block=False)
        except Exception as e:
            logger.error("TTS playback error: %s", e)
        finally:
            if return_to_idle:
                self.state_mgr.set_state(AgentState.IDLE, reason="TTS queued")

    def _speak_response_message(self, message: str) -> None:
        """Speak a goal-level message (used by GoalExecutor)."""
        try:
            self._speak_response(message, CommandMetrics(command_id="goal"))
        except Exception:
            pass

    def _speak_step_progress(self, step, memory) -> None:
        """Optional per-step progress speech."""
        try:
            total = len(memory.plan.steps)
            idx = memory.current_step_index
            self._speak_response_message(f"Step {idx} of {total} done.")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _finalize_result(
        self,
        cid: str,
        transcription: str,
        intent: str,
        tool: Optional[str],
        success: bool,
        verified: bool,
        response_text: str,
        metrics: CommandMetrics,
        lifecycle: Optional[CommandLifecycle] = None,
    ) -> PipelineResult:
        metrics.success = success and verified
        if not getattr(metrics, "selected_tool", None):
            metrics.selected_tool = tool or ""
        # Honest resource snapshot (0.0 when monitoring is unavailable)
        try:
            monitor = getattr(self, "monitor", None)
            if monitor is not None and hasattr(monitor, "get_snapshot"):
                snap = monitor.get_snapshot()
                metrics.ram_used_mb = float(getattr(snap, "ram_used_mb", 0.0) or 0.0)
                metrics.vram_used_mb = float(getattr(snap, "vram_used_mb", 0.0) or 0.0)
        except Exception:
            pass
        self.tracker.record(metrics)
        self.state_mgr.set_state(AgentState.IDLE, reason="Pipeline complete")

        if lifecycle:
            if success and verified:
                lifecycle.succeed(detail=response_text)
            else:
                lifecycle.fail(reason=response_text)

        if self.status_broadcaster:
            if success and verified:
                self.status_broadcaster.broadcast(
                    state="SUCCESS", title="✓ Done", detail=response_text,
                    next_action="Hold Meta + Alt to speak", level="success", command_text=transcription,
                )
            else:
                self.status_broadcaster.broadcast(
                    state="FAILURE", title="✕ Failed", detail=response_text,
                    next_action="Try Meta + Alt again", level="error", command_text=transcription,
                )

        raw_tx = getattr(metrics, "raw_transcription", "") or (lifecycle.raw_transcription if lifecycle else "") or transcription
        norm_tx = getattr(metrics, "transcription", "") or (lifecycle.normalized_transcription if lifecycle else "") or transcription

        return PipelineResult(
            command_id=cid,
            transcription=transcription,
            intent=intent,
            action_executed=tool,
            success=success,
            verified=verified,
            response_text=response_text,
            metrics=metrics,
            raw_transcription=raw_tx,
            normalized_transcription=norm_tx,
        )
