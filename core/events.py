"""
Internal Pub/Sub Event Bus for Laya Desktop Agent.
Decoupled communication between audio, models, tools, monitoring, and UI/logging.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Type, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class Event:
    pass


class EventBus:
    def __init__(self):
        self._subscribers: Dict[Type[Event], List[Callable[[Any], None]]] = {}

    def subscribe(self, event_type: Type[Event], handler: Callable[[Any], None]) -> None:
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        if handler not in self._subscribers[event_type]:
            self._subscribers[event_type].append(handler)

    def publish(self, event: Event) -> None:
        handlers = self._subscribers.get(type(event), [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error("Error dispatching event %s to handler: %s", type(event).__name__, e)
