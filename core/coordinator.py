"""
Pipeline Coordinator for Laya Desktop Agent.
Orchestrates: Audio -> VAD -> STT -> Laya Router -> Executor -> Observer -> Verifier -> Recovery -> TTS.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging
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

        # Autonomous Planning & Speech Subsystems
        self.audio_validator = AudioValidator(sample_rate=config.input.sample_rate)
        catalog = getattr(self.binder, "catalog", None)
        self.speech_corrector = SpeechCorrector(catalog=catalog)
        self.planner = GoalPlanner(AutoPlannerBackend())
        switch_threshold = (
            self.config.planner.method_switch_threshold
            if hasattr(self.config, "planner")
            else 2
        )
        self.goal_executor = GoalExecutor(
            planner=self.planner,
            coordinator=self,
            observer=self.observer,
            verifier=self.verifier,
            method_registry=MethodRegistry(),
            method_switch_threshold=switch_threshold,
        )

    def is_compound_goal(self, text: str) -> bool:
        """Determine if text represents a multi-action goal that needs autonomous decomposition."""
        if not text:
            return False
        goal_cfg = getattr(self.config, "goal", None)
        if goal_cfg and not getattr(goal_cfg, "autonomous_mode", True):
            return False
        clauses = RuleBasedPlannerBackend.SPLIT_REGEX.split(text.strip())
        return len([c for c in clauses if c.strip()]) > 1

    def process_goal(self, goal: str, lifecycle: Optional[CommandLifecycle] = None) -> GoalResult:
        """Direct entry point for executing autonomous goals."""
        return self.goal_executor.execute_goal(goal, lifecycle=lifecycle)

    def process_audio(self, raw_pcm: bytes, lifecycle: Optional[CommandLifecycle] = None) -> Optional[PipelineResult]:

        """Process recorded microphone audio through the full pipeline with lifecycle instrumentation."""
        command_id = lifecycle.command_id if lifecycle else str(uuid.uuid4())[:8]
        metrics = CommandMetrics(command_id=command_id)
        snapshot = self.monitor.get_snapshot()
        metrics.vram_used_mb = snapshot.vram_used_mb
        metrics.ram_used_mb = snapshot.process_rss_mb

        self.state_mgr.set_state(AgentState.PROCESSING, reason="Audio received")

        # 1. Microphone Signal Quality Analysis & Usability Validation
        val_res = self.audio_validator.validate_and_preprocess(raw_pcm)
        metrics.audio_quality_conf = 1.0 if val_res.is_usable else 0.0
        logger.info(
            "[%s] Signal Validation: usable=%s, RMS=%.1f dBFS, Peak=%.1f dBFS, Reason='%s'",
            command_id,
            val_res.is_usable,
            val_res.rms_dbfs,
            val_res.peak_dbfs,
            val_res.failure_reason,
        )

        if not val_res.is_usable:
            logger.info("[%s] Audio rejected by validator: %s", command_id, val_res.failure_reason)
            if lifecycle:
                lifecycle.fail(reason=val_res.failure_reason)
            self.state_mgr.set_state(AgentState.IDLE, reason=val_res.failure_reason)
            if self.status_broadcaster:
                self.status_broadcaster.broadcast(
                    state="FAILURE",
                    title="● No audio detected",
                    detail=val_res.failure_reason,
                    next_action="Hold Meta + Alt and speak clearly",
                    level="warning",
                )
            self._speak_response("I didn't hear anything.", metrics)
            return None

        # 2. VAD Processing on conditioned audio
        with self.tracker.measure("vad", metrics):
            vad_res = self.vad.process(val_res.preprocessed_pcm)

        if lifecycle:
            lifecycle.transition(CommandStage.VAD_COMPLETE, detail=f"has_speech={vad_res.has_speech}")

        if not vad_res.has_speech:
            logger.info("[%s] No speech detected in audio by VAD. Ignoring.", command_id)
            if lifecycle:
                lifecycle.fail(reason="No speech detected")
            self.state_mgr.set_state(AgentState.IDLE, reason="No speech")
            if self.status_broadcaster:
                self.status_broadcaster.broadcast(
                    state="FAILURE",
                    title="● No speech detected",
                    detail="Audio input had no voiced speech frames",
                    next_action="Hold Meta + Alt and speak clearly",
                    level="warning",
                )
            self._speak_response("No speech detected.", metrics)
            return None

        speech_pcm = vad_res.speech_pcm

        # 3. Faster-Whisper Speech-to-Text with Vocabulary Biasing
        if lifecycle:
            lifecycle.transition(CommandStage.STT_START)

        if self.status_broadcaster:
            self.status_broadcaster.broadcast(
                state="TRANSCRIBING",
                title="◌ Transcribing...",
                detail="Converting speech to text...",
                next_action="",
            )

        # Generate domain vocabulary prompt and hotwords from AppCatalog
        whisper_prompt = None
        whisper_hotwords = None
        if hasattr(self.binder, "catalog") and self.binder.catalog:
            try:
                whisper_prompt = self.binder.catalog.generate_whisper_prompt()
                whisper_hotwords = self.binder.catalog.generate_hotwords()
            except Exception as e:
                logger.debug("[%s] Could not generate whisper vocabulary prompt: %s", command_id, e)

        with self.tracker.measure("stt", metrics):
            trans_res = self.model_mgr.whisper.transcribe(
                speech_pcm,
                initial_prompt=whisper_prompt,
                hotwords=whisper_hotwords,
                beam_size=5,
            )
            transcription = trans_res.text if hasattr(trans_res, "text") else str(trans_res[0])
            stt_conf = getattr(trans_res, "confidence", 0.95)

        if lifecycle:
            lifecycle.transition(CommandStage.STT_COMPLETE, detail=f"conf={stt_conf:.2f}")

        raw_tx = getattr(trans_res, "raw_text", transcription)
        metrics.raw_transcription = raw_tx
        metrics.stt_confidence = stt_conf

        if not transcription.strip():
            logger.info("[%s] Speech transcription was empty.", command_id)
            if lifecycle:
                lifecycle.fail(reason="Empty transcription")
            self.state_mgr.set_state(AgentState.IDLE, reason="Empty transcription")
            if self.status_broadcaster:
                self.status_broadcaster.broadcast(
                    state="FAILURE",
                    title="● Could not understand",
                    detail="No words recognized in audio",
                    next_action="Hold Meta + Alt and speak clearly",
                    level="warning",
                )
            self._speak_response("I couldn't understand that.", metrics)
            return None

        # 4. Context-Aware Speech Correction & Intent Understanding
        pre_obs_for_correction = self.observer.observe(delay_sec=0.0)
        raw_tx, normalized_tx = self.speech_corrector.normalize(
            transcription, desktop_state=pre_obs_for_correction
        )
        if normalized_tx != transcription:
            logger.info("[%s] Speech transcription normalized: '%s' -> '%s'", command_id, transcription, normalized_tx)

        metrics.raw_transcription = raw_tx
        metrics.transcription = normalized_tx

        logger.info("[%s] Transcribed: '%s' -> '%s' (conf: %.2f)", command_id, raw_tx, normalized_tx, stt_conf)

        if lifecycle:
            lifecycle.raw_transcription = raw_tx
            lifecycle.normalized_transcription = normalized_tx
            lifecycle.transcription = normalized_tx
            lifecycle.transition(CommandStage.TRANSCRIPT_READY, detail=normalized_tx)

        # Broadcast the heard speech immediately to the transient UI
        if self.status_broadcaster:
            self.status_broadcaster.broadcast(
                state="UNDERSTANDING",
                title=f"Heard: \"{normalized_tx}\"",
                detail="Resolving desktop command...",
                next_action="",
                command_text=normalized_tx,
            )

        return self.process_text_command(normalized_tx, metrics=metrics, command_id=command_id, lifecycle=lifecycle)

    def process_text_command(
        self,
        command_text: str,
        metrics: Optional[CommandMetrics] = None,
        command_id: Optional[str] = None,
        lifecycle: Optional[CommandLifecycle] = None,
        allow_autonomous_decomposition: bool = True,
    ) -> PipelineResult:
        """Process a text command directly (e.g. from CLI or post-STT) with full stage tracking."""
        cid = (lifecycle.command_id if lifecycle else None) or command_id or str(uuid.uuid4())[:8]
        m = metrics or CommandMetrics(command_id=cid, transcription=command_text)

        # Ensure lifecycle exists for tracking and watchdog
        if lifecycle is None:
            if self.lifecycle_mgr:
                lifecycle = self.lifecycle_mgr.new_command()
                cid = lifecycle.command_id
            else:
                lifecycle = CommandLifecycle(command_id=cid)

        # Watchdog timeout callback (Requirement 9)
        def _on_cmd_timeout(lc: CommandLifecycle):
            logger.error("[%s] Command watchdog expired after timeout! Stuck at: %s", cid, lc.current_stage.value)
            self.state_mgr.set_state(AgentState.IDLE, reason="Command watchdog timeout")
            if self.status_broadcaster:
                self.status_broadcaster.broadcast(
                    state="FAILURE",
                    title="✕ Timed Out",
                    detail=f"Stuck at: {lc.current_stage.value}",
                    next_action="Hold Meta + Alt to try again",
                    level="error",
                    command_text=command_text,
                )

        if not lifecycle.is_finalized:
            lifecycle.start_watchdog(timeout_sec=16.0, on_timeout=_on_cmd_timeout)

        try:
            # Autonomous multi-step goal routing
            if allow_autonomous_decomposition and self.is_compound_goal(command_text):
                logger.info("[%s] Multi-step compound goal detected: '%s'. Delegating to GoalExecutor.", cid, command_text)
                lifecycle.transition(CommandStage.PLANNING, detail=f"Goal: {command_text}")
                goal_res = self.goal_executor.execute_goal(command_text, lifecycle=lifecycle)
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

            # 3. Laya Decision & Routing
            lifecycle.transition(CommandStage.LAYA_START)
            self.state_mgr.set_state(AgentState.PROCESSING, reason="Laya routing")
            if self.status_broadcaster:
                self.status_broadcaster.broadcast(
                    state="UNDERSTANDING",
                    title="◌ Understanding...",
                    detail=f"'{command_text}'",
                    next_action="",
                    command_text=command_text,
                )

            with self.tracker.measure("laya_routing", m):
                lower = command_text.lower().strip()
                # Fast-path explicit prefixes to avoid 3.0s ModernBERT inference/timeout on CPU
                if any(lower.startswith(p) for p in ["open ", "launch ", "start "]):
                    decision = LayaDecision(
                        intent="open_application",
                        confidence=0.98,
                        safety_level="safe",
                        requires_confirmation=False,
                        requires_observation=True,
                        raw_decision={"fast_path": True},
                    )
                elif any(lower.startswith(p) for p in ["switch to ", "focus ", "bring up ", "go to "]):
                    decision = LayaDecision(
                        intent="switch_window",
                        confidence=0.98,
                        safety_level="safe",
                        requires_confirmation=False,
                        requires_observation=True,
                        raw_decision={"fast_path": True},
                    )
                elif any(lower.startswith(p) for p in ["close ", "quit ", "exit ", "close this"]):
                    decision = LayaDecision(
                        intent="close_application",
                        confidence=0.98,
                        safety_level="safe",
                        requires_confirmation=False,
                        requires_observation=True,
                        raw_decision={"fast_path": True},
                    )
                else:
                    pre_obs_pre = self.observer.observe(delay_sec=0.0)
                    context = {
                        "active_window": pre_obs_pre.active_window_name,
                        "active_class": pre_obs_pre.active_window_class,
                        "desktop": pre_obs_pre.desktop_number,
                    }
                    try:
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                            fut = pool.submit(self.model_mgr.laya.decide, command_text, context=context)
                            decision = fut.result(timeout=3.0)
                    except Exception as e:
                        logger.warning("[%s] Laya router timeout or error (%s). Falling back to heuristic router.", cid, e)
                        decision = self.model_mgr.laya._heuristic_fallback(command_text, 0.0)

            m.intent = decision.intent
            lifecycle.intent = decision.intent
            lifecycle.transition(CommandStage.LAYA_COMPLETE, detail=f"intent={decision.intent} (conf: {decision.confidence:.2f})")

            logger.info("[%s] Laya routed intent: %s (conf: %.2f)", cid, decision.intent, decision.confidence)

            # 4. Parameter Binding
            is_explicit = any(lower.startswith(p) for p in [
                "open ", "launch ", "start ", "close ", "quit ", "exit ", "close this",
                "switch to ", "focus ", "bring up ", "go to ",
                "scroll ", "type ", "press ", "take a screenshot", "screenshot",
                "mute", "volume ", "maximize", "minimize", "fullscreen"
            ])
            if not is_explicit and decision.confidence < self.config.laya.confidence_threshold:
                logger.info("[%s] Ignoring low-confidence intent %s (conf: %.2f < %.2f) for '%s'", cid, decision.intent, decision.confidence, self.config.laya.confidence_threshold, command_text)
                response_text = f"I didn't understand: '{command_text}'"
                lifecycle.fail(reason=response_text)
                return self._finalize_result(cid, command_text, decision.intent, None, False, False, response_text, m, lifecycle=lifecycle)

            bound_action = self.binder.bind(command_text, decision)
            if not bound_action:
                response_text = f"I couldn't find an application matching '{command_text}'."
                logger.info("[%s] %s", cid, response_text)
                self._speak_response(response_text, m)
                lifecycle.fail(reason=response_text)
                return self._finalize_result(cid, command_text, decision.intent, None, False, False, response_text, m, lifecycle=lifecycle)

            m.selected_tool = bound_action.tool_name
            m.target = bound_action.target or bound_action.tool_name
            m.entity_resolution_conf = getattr(bound_action, "confidence", 0.95)
            m.intent_confidence = decision.confidence

            verb = (
                "Opening" if bound_action.tool_name == "open_application"
                else ("Closing" if bound_action.tool_name == "close_application"
                else ("Switching to" if bound_action.tool_name == "switch_window" else "Executing"))
            )

            lifecycle.target = m.target
            lifecycle.transition(CommandStage.ENTITY_READY, detail=f"{verb} {m.target}")

            # Handle clarification if user request was ambiguous
            if bound_action.tool_name == "clarify":
                prompt = bound_action.arguments.get("prompt", f"Could you clarify your request: '{command_text}'?")
                logger.info("[%s] Clarification requested: %s", cid, prompt)
                self._speak_response(prompt, m)
                lifecycle.transition(CommandStage.PLAN_READY, detail="clarify")
                if self.status_broadcaster:
                    self.status_broadcaster.broadcast(
                        state="CLARIFY",
                        title="? Clarification Needed",
                        detail=prompt,
                        next_action="Hold Meta + Alt to answer",
                        level="warning",
                        command_text=command_text,
                    )
                return self._finalize_result(cid, command_text, decision.intent, "clarify", True, True, prompt, m, lifecycle=lifecycle)

            # 5. Diagnostic Mode Handling (zero execution)
            if self.diagnostic_mode:
                response_text = f"[DIAGNOSTIC] Intent: {decision.intent} | Tool: {bound_action.tool_name} | Target: {m.target}"
                logger.info("[%s] Diagnostic mode: Action not executed.", cid)
                lifecycle.succeed(detail=response_text)
                return self._finalize_result(cid, command_text, decision.intent, bound_action.tool_name, True, True, response_text, m, lifecycle=lifecycle)

            # 6. Safety Confirmation Check
            if decision.requires_confirmation:
                response_text = f"Action '{bound_action.tool_name}' requires confirmation. Execution blocked for safety."
                logger.warning("[%s] %s", cid, response_text)
                self._speak_response(response_text, m)
                lifecycle.fail(reason=response_text)
                return self._finalize_result(cid, command_text, decision.intent, bound_action.tool_name, False, False, response_text, m, lifecycle=lifecycle)

            logger.info(
                "[%s] PIPELINE RESOLUTION: HEARD='%s' -> TARGET='%s' (tool=%s | conf: STT=%.2f, Res=%.2f, Intent=%.2f)",
                cid, command_text, m.target, bound_action.tool_name, m.stt_confidence, m.entity_resolution_conf, m.intent_confidence
            )

            # Pre-action observation
            pre_obs = self.observer.observe(delay_sec=0.0)

            # 7. Action Execution
            lifecycle.transition(CommandStage.EXECUTION_START, detail=f"{verb} {m.target}")
            self.state_mgr.set_state(AgentState.EXECUTING, reason=bound_action.tool_name)
            if self.status_broadcaster:
                self.status_broadcaster.broadcast(
                    state="EXECUTING",
                    title=f"{verb}: {m.target}",
                    detail=f"Heard: \"{command_text}\"",
                    next_action="",
                    command_text=command_text,
                )

            with self.tracker.measure("execution", m):
                tool_res = self.executor.execute_action(bound_action)

            lifecycle.transition(CommandStage.EXECUTION_COMPLETE, detail=f"success={tool_res.success}")

            # 8. Observation & Verification
            lifecycle.transition(CommandStage.VERIFICATION_START)
            self.state_mgr.set_state(AgentState.VERIFYING, reason="Observing state")
            if self.status_broadcaster:
                self.status_broadcaster.broadcast(
                    state="VERIFYING",
                    title="◌ Verifying...",
                    detail=f"Checking {m.target} active window...",
                    next_action="",
                    command_text=command_text,
                )

            with self.tracker.measure("verification", m):
                obs_delay = 0.05 if bound_action.tool_name in ("open_application", "switch_window") else 0.0
                post_obs = self.observer.observe(delay_sec=obs_delay)
                verify_res = self.verifier.verify(bound_action, tool_res, pre_obs, post_obs)

            # 9. Recovery if unverified
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

            # 10. Text-to-Speech Feedback
            response_text = tool_res.message
            self._speak_response(response_text, m)

            # 11. Cache cleanup pass
            self.reaper.reap_all()

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
                    state="FAILURE",
                    title="✕ Error",
                    detail=f"{str(e)[:40]}",
                    next_action="Hold Meta + Alt to try again",
                    level="error",
                    command_text=command_text,
                )
            return self._finalize_result(
                cid=cid,
                transcription=command_text,
                intent="error",
                tool=None,
                success=False,
                verified=False,
                response_text=f"An error occurred: {e}",
                metrics=m,
                lifecycle=lifecycle,
            )

        finally:
            lifecycle.stop_watchdog()
            self.state_mgr.set_state(AgentState.IDLE, reason="Command processing ended")

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
                    state="SUCCESS",
                    title="✓ Done",
                    detail=response_text,
                    next_action="Hold Meta + Alt to speak",
                    level="success",
                    command_text=transcription,
                )
            else:
                self.status_broadcaster.broadcast(
                    state="FAILURE",
                    title="✕ Failed",
                    detail=response_text,
                    next_action="Try Meta + Alt again",
                    level="error",
                    command_text=transcription,
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
