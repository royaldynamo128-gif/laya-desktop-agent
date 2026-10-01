"""
Unified Command Lifecycle & Watchdog Engine for Laya Desktop Agent.
Enforces request IDs, timestamped stage transitions, per-command timeouts,
and structured diagnostic tracking to prevent indefinite 'Processing...' stalls.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
import time
from typing import Callable, Deque, Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class CommandStage(str, Enum):
    IDLE = "IDLE"
    PTT_START = "PTT_START"
    AUDIO_START = "AUDIO_START"
    AUDIO_STOP = "AUDIO_STOP"
    VAD_COMPLETE = "VAD_COMPLETE"
    STT_START = "STT_START"
    STT_COMPLETE = "STT_COMPLETE"
    TRANSCRIPT_READY = "TRANSCRIPT_READY"
    LAYA_START = "LAYA_START"
    LAYA_COMPLETE = "LAYA_COMPLETE"
    INTENT_READY = "INTENT_READY"
    ENTITY_READY = "ENTITY_READY"
    PLAN_READY = "PLAN_READY"
    PLANNING = "PLANNING"
    STEP_START = "STEP_START"
    STEP_COMPLETE = "STEP_COMPLETE"
    REPLANNING = "REPLANNING"
    GOAL_VERIFIED = "GOAL_VERIFIED"
    EXECUTION_START = "EXECUTION_START"
    EXECUTION_COMPLETE = "EXECUTION_COMPLETE"
    VERIFICATION_START = "VERIFICATION_START"
    VERIFICATION_COMPLETE = "VERIFICATION_COMPLETE"
    RESPONSE_READY = "RESPONSE_READY"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"


STAGE_USER_LABELS = {
    CommandStage.IDLE: "",
    CommandStage.PTT_START: "🎙 Listening",
    CommandStage.AUDIO_START: "🎙 Listening",
    CommandStage.AUDIO_STOP: "◌ Finalizing audio...",
    CommandStage.VAD_COMPLETE: "◌ Analyzing speech...",
    CommandStage.STT_START: "◌ Transcribing...",
    CommandStage.STT_COMPLETE: "◌ Transcription complete",
    CommandStage.TRANSCRIPT_READY: "Heard speech",
    CommandStage.LAYA_START: "◌ Understanding...",
    CommandStage.LAYA_COMPLETE: "◌ Understanding complete",
    CommandStage.INTENT_READY: "◌ Planning action...",
    CommandStage.ENTITY_READY: "◌ Resolved target",
    CommandStage.PLAN_READY: "◌ Ready to execute",
    CommandStage.PLANNING: "◌ Planning goal steps...",
    CommandStage.STEP_START: "◌ Executing step...",
    CommandStage.STEP_COMPLETE: "◌ Step complete",
    CommandStage.REPLANNING: "◌ Replanning alternative method...",
    CommandStage.GOAL_VERIFIED: "✓ Goal Verified",
    CommandStage.EXECUTION_START: "◌ Executing...",
    CommandStage.EXECUTION_COMPLETE: "◌ Execution complete",
    CommandStage.VERIFICATION_START: "◌ Verifying...",
    CommandStage.VERIFICATION_COMPLETE: "◌ Verification complete",
    CommandStage.RESPONSE_READY: "◌ Responding...",
    CommandStage.SUCCESS: "✓ Done",
    CommandStage.FAILED: "✕ Failed",
    CommandStage.CANCELLED: "⏹ Cancelled",
    CommandStage.TIMEOUT: "✕ Timed Out",
}



STAGE_TIMEOUTS = {
    CommandStage.AUDIO_START: 30.0,
    CommandStage.VAD_COMPLETE: 1.5,
    CommandStage.STT_START: 6.0,
    CommandStage.LAYA_START: 3.5,
    CommandStage.PLANNING: 3.5,
    CommandStage.EXECUTION_START: 8.0,
    CommandStage.VERIFICATION_START: 3.0,
}

STAGE_RECOVERY_PATHS = {
    CommandStage.AUDIO_START: "Auto-cancel recording on max duration limit",
    CommandStage.VAD_COMPLETE: "Reject with 'No speech detected' and prompt user to hold Win+Alt",
    CommandStage.STT_START: "Fallback to CPU int8 or reject with 'I couldn't understand that'",
    CommandStage.LAYA_START: "Fast fallback to deterministic keyword router",
    CommandStage.PLANNING: "Fallback to single-action execution",
    CommandStage.EXECUTION_START: "Retry action via window rescue or alternate launch path",
    CommandStage.VERIFICATION_START: "Poll active window or initiate recovery sequence",
}


@dataclass
class StageEntry:
    stage: CommandStage
    timestamp: float
    elapsed_ms: float
    detail: str = ""
    timeout_sec: float = 0.0
    recovery_path: str = ""


class CommandLifecycle:
    def __init__(self, command_id: str, on_ui_update: Optional[Callable[[str, str, str, str], None]] = None):
        self.command_id = command_id
        self.on_ui_update = on_ui_update
        self.created_at = time.time()
        self.current_stage = CommandStage.IDLE
        self.history: List[StageEntry] = []
        self.stage_timestamps: Dict[CommandStage, float] = {}
        self.transcription = ""
        self.raw_transcription = ""
        self.normalized_transcription = ""
        self.target = ""
        self.intent = ""
        self.error_message: Optional[str] = None
        self._watchdog_timer: Optional[threading.Timer] = None
        self._watchdog_lock = threading.RLock()
        self._is_finalized = False

    @property
    def total_elapsed_ms(self) -> float:
        return (time.time() - self.created_at) * 1000.0

    @property
    def is_finalized(self) -> bool:
        return self._is_finalized

    def transition(
        self,
        stage: CommandStage,
        detail: str = "",
        timeout_sec: Optional[float] = None,
        recovery_path: Optional[str] = None,
    ) -> None:
        """Record stage transition with timestamp and log progression."""
        if self._is_finalized and stage not in (CommandStage.SUCCESS, CommandStage.FAILED, CommandStage.TIMEOUT, CommandStage.CANCELLED):
            return

        now = time.time()
        elapsed = (now - self.created_at) * 1000.0
        self.current_stage = stage
        self.stage_timestamps[stage] = now

        eff_timeout = timeout_sec if timeout_sec is not None else STAGE_TIMEOUTS.get(stage, 0.0)
        eff_recovery = recovery_path if recovery_path is not None else STAGE_RECOVERY_PATHS.get(stage, "")

        entry = StageEntry(
            stage=stage,
            timestamp=now,
            elapsed_ms=elapsed,
            detail=detail,
            timeout_sec=eff_timeout,
            recovery_path=eff_recovery,
        )
        self.history.append(entry)

        logger.info(
            "[%s] STAGE: %s (+%.1fms)%s%s",
            self.command_id,
            stage.value,
            elapsed,
            f" | {detail}" if detail else "",
            f" [timeout: {eff_timeout}s]" if eff_timeout > 0 else "",
        )

        if stage in (CommandStage.SUCCESS, CommandStage.FAILED, CommandStage.TIMEOUT, CommandStage.CANCELLED):
            self._is_finalized = True
            self.stop_watchdog()

    def start_watchdog(self, timeout_sec: float, on_timeout: Callable[["CommandLifecycle"], None]) -> None:
        """Start background watchdog timer for the command."""
        with self._watchdog_lock:
            self.stop_watchdog()
            if self._is_finalized:
                return

            def _fire():
                with self._watchdog_lock:
                    if not self._is_finalized:
                        logger.error(
                            "[%s] COMMAND WATCHDOG EXPIRED after %.1fs! Stuck at stage: %s",
                            self.command_id,
                            timeout_sec,
                            self.current_stage.value,
                        )
                        self.error_message = f"Command timed out after {timeout_sec:.1f}s at stage: {self.current_stage.value}"
                        self.transition(CommandStage.TIMEOUT, detail=self.error_message)
                        try:
                            on_timeout(self)
                        except Exception as e:
                            logger.error("[%s] Error in watchdog on_timeout callback: %s", self.command_id, e)

            self._watchdog_timer = threading.Timer(timeout_sec, _fire)
            self._watchdog_timer.daemon = True
            self._watchdog_timer.name = f"Watchdog-{self.command_id}"
            self._watchdog_timer.start()

    def stop_watchdog(self) -> None:
        """Cancel active watchdog timer."""
        with self._watchdog_lock:
            if self._watchdog_timer is not None:
                self._watchdog_timer.cancel()
                self._watchdog_timer = None

    def succeed(self, detail: str = "") -> None:
        self.transition(CommandStage.SUCCESS, detail=detail)

    def fail(self, reason: str) -> None:
        self.error_message = reason
        self.transition(CommandStage.FAILED, detail=reason)

    def cancel(self, reason: str = "User cancelled") -> None:
        self.transition(CommandStage.CANCELLED, detail=reason)

    def to_diagnostic_dict(self) -> Dict[str, Any]:
        return {
            "command_id": self.command_id,
            "created_at": time.strftime("%H:%M:%S", time.localtime(self.created_at)),
            "total_ms": round(self.total_elapsed_ms, 1),
            "final_stage": self.current_stage.value,
            "transcription": self.transcription,
            "intent": self.intent,
            "target": self.target,
            "error": self.error_message,
            "stages": [
                {
                    "stage": e.stage.value,
                    "elapsed_ms": round(e.elapsed_ms, 1),
                    "detail": e.detail,
                }
                for e in self.history
            ],
        }


class LifecycleManager:
    """Thread-safe lifecycle coordinator maintaining active command tracking and history ledger."""

    def __init__(self, history_size: int = 10):
        self._history: Deque[CommandLifecycle] = deque(maxlen=history_size)
        self._counter = 0
        self._lock = threading.Lock()
        self._active_lifecycle: Optional[CommandLifecycle] = None

    def new_command(self) -> CommandLifecycle:
        """Create and track a new unique command lifecycle."""
        with self._lock:
            self._counter += 1
            cid = f"cmd-{self._counter:03d}"
            lifecycle = CommandLifecycle(command_id=cid)
            self._active_lifecycle = lifecycle
            self._history.append(lifecycle)
            return lifecycle

    @property
    def active_command(self) -> Optional[CommandLifecycle]:
        with self._lock:
            return self._active_lifecycle

    def clear_active(self, lifecycle: Optional[CommandLifecycle] = None) -> None:
        with self._lock:
            if lifecycle is None or self._active_lifecycle is lifecycle:
                self._active_lifecycle = None

    def get_last_command(self) -> Optional[CommandLifecycle]:
        with self._lock:
            if self._history:
                return self._history[-1]
            return None

    def get_all_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [cmd.to_diagnostic_dict() for cmd in self._history]
