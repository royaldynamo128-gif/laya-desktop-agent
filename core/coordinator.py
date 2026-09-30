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
from audio.player import AudioPlayer
from models.manager import ModelManager
from execution.parameter_binder import ParameterBinder, BoundAction
from execution.executor import ActionExecutor
from verification.observer import DesktopObserver, DesktopObservation
from verification.verifier import ActionVerifier, VerificationResult
from verification.recovery import RecoveryManager
from monitoring.latency_tracker import LatencyTracker, CommandMetrics
from monitoring.cache_reaper import CacheReaper
from monitoring.resource_monitor import ResourceMonitor

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
        diagnostic_mode: bool = False,
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
        self.diagnostic_mode = diagnostic_mode

    def process_audio(self, raw_pcm: bytes) -> Optional[PipelineResult]:
        """Process recorded microphone audio through the full pipeline."""
        command_id = str(uuid.uuid4())[:8]
        metrics = CommandMetrics(command_id=command_id)
        snapshot = self.monitor.get_snapshot()
        metrics.vram_used_mb = snapshot.vram_used_mb
        metrics.ram_used_mb = snapshot.process_rss_mb

        self.state_mgr.set_state(AgentState.PROCESSING, reason="Audio received")

        # 1. VAD Processing
        with self.tracker.measure("vad", metrics):
            vad_res = self.vad.process(raw_pcm)

        if not vad_res.has_speech:
            logger.info("No speech detected in audio. Ignoring.")
            self.state_mgr.set_state(AgentState.IDLE, reason="No speech")
            return None

        # 2. Faster-Whisper Speech-to-Text
        with self.tracker.measure("stt", metrics):
            transcription, _ = self.model_mgr.whisper.transcribe(vad_res.speech_pcm)

        metrics.transcription = transcription
        if not transcription.strip():
            logger.info("Speech transcription was empty.")
            self.state_mgr.set_state(AgentState.IDLE, reason="Empty transcription")
            return None

        logger.info("Command [%s] Transcribed: '%s'", command_id, transcription)

        return self.process_text_command(transcription, metrics=metrics, command_id=command_id)

    def process_text_command(
        self,
        command_text: str,
        metrics: Optional[CommandMetrics] = None,
        command_id: Optional[str] = None,
    ) -> PipelineResult:
        """Process a text command directly (e.g. from CLI or post-STT)."""
        cid = command_id or str(uuid.uuid4())[:8]
        m = metrics or CommandMetrics(command_id=cid, transcription=command_text)

        # 3. Laya Decision & Routing
        self.state_mgr.set_state(AgentState.PROCESSING, reason="Laya routing")
        with self.tracker.measure("laya_routing", m):
            pre_obs = self.observer.observe(delay_sec=0.0)
            context = {
                "active_window": pre_obs.active_window_name,
                "active_class": pre_obs.active_window_class,
                "desktop": pre_obs.desktop_number,
            }
            decision = self.model_mgr.laya.decide(command_text, context=context)

        m.intent = decision.intent
        logger.info("Laya routed command to intent: %s (conf: %.2f)", decision.intent, decision.confidence)

        # 4. Parameter Binding
        bound_action = self.binder.bind(command_text, decision)
        if not bound_action:
            response_text = f"I'm not sure how to handle that request: {command_text}"
            return self._finalize_result(cid, command_text, decision.intent, None, False, False, response_text, m)

        m.selected_tool = bound_action.tool_name

        # 5. Diagnostic Mode Handling (zero execution)
        if self.diagnostic_mode:
            response_text = f"[DIAGNOSTIC] Intent: {decision.intent} | Tool: {bound_action.tool_name} | Args: {bound_action.arguments}"
            logger.info("Diagnostic mode: Action not executed.")
            return self._finalize_result(cid, command_text, decision.intent, bound_action.tool_name, True, True, response_text, m)

        # 6. Safety Confirmation Check
        if decision.requires_confirmation:
            response_text = f"Action '{bound_action.tool_name}' requires confirmation. Execution blocked for safety."
            logger.warning(response_text)
            self._speak_response(response_text, m)
            return self._finalize_result(cid, command_text, decision.intent, bound_action.tool_name, False, False, response_text, m)

        # 7. Action Execution
        self.state_mgr.set_state(AgentState.EXECUTING, reason=bound_action.tool_name)
        with self.tracker.measure("execution", m):
            tool_res = self.executor.execute_action(bound_action)

        # 8. Observation & Verification
        self.state_mgr.set_state(AgentState.VERIFYING, reason="Observing state")
        with self.tracker.measure("verification", m):
            post_obs = self.observer.observe(delay_sec=0.3)
            verify_res = self.verifier.verify(bound_action, tool_res, pre_obs, post_obs)

        # 9. Recovery if unverified
        if not verify_res.verified:
            with self.tracker.measure("recovery", m):
                recovery_res, final_verify = self.recovery_mgr.attempt_recovery(
                    action=bound_action,
                    verification=verify_res,
                    execute_fn=lambda a: self.executor.execute_action(a),
                    verify_fn=lambda a, r: self.verifier.verify(a, r, pre_obs, self.observer.observe(0.2)),
                )
                if recovery_res:
                    tool_res = recovery_res
                verify_res = final_verify

        # 10. Text-to-Speech Feedback
        response_text = tool_res.message
        self._speak_response(response_text, m)

        # 11. Cache cleanup pass
        self.reaper.reap_all()

        return self._finalize_result(
            cid, command_text, decision.intent, bound_action.tool_name,
            tool_res.success, verify_res.verified, response_text, m
        )

    def _speak_response(self, text: str, metrics: CommandMetrics) -> None:
        """Synthesize and play audio response via Piper on CPU."""
        if not self.config.tts.enabled or not text:
            return

        self.state_mgr.set_state(AgentState.SPEAKING, reason="Playing audio response")
        try:
            with self.tracker.measure("tts", metrics):
                wav_path = self.model_mgr.piper.synthesize_to_wav(text)

            if wav_path and wav_path.exists():
                self.player.play_wav(wav_path, block=False)
        except Exception as e:
            logger.error("TTS playback error: %s", e)

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
    ) -> PipelineResult:
        metrics.success = success and verified
        self.tracker.record(metrics)
        self.state_mgr.set_state(AgentState.IDLE, reason="Pipeline complete")

        return PipelineResult(
            command_id=cid,
            transcription=transcription,
            intent=intent,
            action_executed=tool,
            success=success,
            verified=verified,
            response_text=response_text,
            metrics=metrics,
        )
