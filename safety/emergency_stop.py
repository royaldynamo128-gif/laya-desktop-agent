"""
Global Emergency Stop Handler for Laya Desktop Agent.
Instant hardware-latched abort mechanism. Halts tool execution, clears queues, and mutes speech.
"""

from typing import Callable, List
import logging
import threading

logger = logging.getLogger(__name__)


class EmergencyStop:
    def __init__(self):
        self._stopped = False
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[], None]] = []

    def is_stopped(self) -> bool:
        with self._lock:
            return self._stopped

    def trigger(self, reason: str = "Emergency Stop Triggered") -> None:
        """Instantly trip the emergency stop latch and invoke cancellation handlers."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True

        logger.critical("!!! EMERGENCY STOP ACTIVATED: %s !!!", reason)

        # Fire callbacks outside lock
        for cb in self._callbacks:
            try:
                cb()
            except Exception as e:
                logger.error("Error executing emergency stop callback: %s", e)

    def reset(self) -> None:
        """Reset emergency stop latch, allowing normal agent execution to resume."""
        with self._lock:
            self._stopped = False
        logger.info("Emergency Stop latch cleared. Agent resumed normal operation.")

    def register_callback(self, callback: Callable[[], None]) -> None:
        """Register a cleanup/abort callback (e.g. mute TTS, cancel current tool)."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)
