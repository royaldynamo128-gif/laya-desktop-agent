"""
Agent State Machine for Laya Desktop Agent.
Thread-safe lifecycle state management and transition callbacks.
"""

from enum import Enum
from typing import Callable, List
import logging
import threading

logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    SPEAKING = "SPEAKING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"


class StateManager:
    def __init__(self, initial_state: AgentState = AgentState.IDLE):
        self._state = initial_state
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[AgentState, AgentState], None]] = []

    @property
    def current_state(self) -> AgentState:
        with self._lock:
            return self._state

    def set_state(self, new_state: AgentState, reason: str = "") -> None:
        old_state = None
        with self._lock:
            if self._state == new_state:
                return
            old_state = self._state
            self._state = new_state

        logger.debug("State transition: %s -> %s %s", old_state.value, new_state.value, f"({reason})" if reason else "")

        for cb in self._callbacks:
            try:
                cb(old_state, new_state)
            except Exception as e:
                logger.debug("Error in state callback: %s", e)

    def register_callback(self, callback: Callable[[AgentState, AgentState], None]) -> None:
        if callback not in self._callbacks:
            self._callbacks.append(callback)
