"""
Desktop State Observer for Laya Desktop Agent.
Captures post-action desktop state using kdotool and AT-SPI2 without visual overhead.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import logging
import time
from integrations.kdotool_client import KdotoolClient
from integrations.atspi_client import AtspiClient

logger = logging.getLogger(__name__)


@dataclass
class DesktopObservation:
    timestamp: float
    active_window_id: Optional[str]
    active_window_name: str
    active_window_class: str
    desktop_number: int
    open_window_ids: List[str]


class DesktopObserver:
    def __init__(self, kdotool: KdotoolClient, atspi: AtspiClient):
        self.kdotool = kdotool
        self.atspi = atspi

    def observe(self, delay_sec: float = 0.3) -> DesktopObservation:
        """Capture current desktop state after giving the desktop a brief moment to update."""
        if delay_sec > 0:
            time.sleep(delay_sec)

        active_id = self.kdotool.get_active_window()
        active_name = self.kdotool.get_window_name(active_id) if active_id else ""
        active_class = self.kdotool.get_window_class(active_id) if active_id else ""
        desktop_num = self.kdotool.get_desktop()
        open_wins = self.kdotool.search("", limit=15)

        obs = DesktopObservation(
            timestamp=time.time(),
            active_window_id=active_id,
            active_window_name=active_name,
            active_window_class=active_class,
            desktop_number=desktop_num,
            open_window_ids=open_wins,
        )

        logger.debug(
            "Observation captured: Active=[%s] '%s' (class: %s) | Open count: %d",
            active_id,
            active_name,
            active_class,
            len(open_wins),
        )
        return obs
