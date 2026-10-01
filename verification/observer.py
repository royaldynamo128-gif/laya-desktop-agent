"""
Desktop State Observer for Laya Desktop Agent.
Captures post-action desktop state using kdotool and AT-SPI2 without visual overhead.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging
import time
from integrations.kdotool_client import KdotoolClient
from integrations.atspi_client import AtspiClient

logger = logging.getLogger(__name__)


@dataclass
class WindowInfo:
    window_id: str
    name: str = ""
    wm_class: str = ""
    is_active: bool = False


@dataclass
class DesktopObservation:
    timestamp: float
    active_window_id: Optional[str]
    active_window_name: str
    active_window_class: str
    desktop_number: int
    open_window_ids: List[str]
    # Enhanced observational fields
    active_window_title: str = ""
    all_windows: List[WindowInfo] = field(default_factory=list)
    focused_ui_elements: List[str] = field(default_factory=list)
    ui_summary: str = ""

    def __post_init__(self):
        if not self.active_window_title:
            self.active_window_title = self.active_window_name


class DesktopObserver:
    def __init__(self, kdotool: KdotoolClient, atspi: AtspiClient):
        self.kdotool = kdotool
        self.atspi = atspi

    def observe(self, delay_sec: float = 0.0) -> DesktopObservation:
        """Capture current desktop state without unnecessary sleep delay."""
        if delay_sec > 0:
            time.sleep(delay_sec)

        active_id = self.kdotool.get_active_window()
        active_name = self.kdotool.get_window_name(active_id) if active_id else ""
        active_class = self.kdotool.get_window_class(active_id) if active_id else ""
        desktop_num = self.kdotool.get_desktop()
        open_wins = self.kdotool.search("", limit=15)

        # Build window info objects
        windows: List[WindowInfo] = []
        for wid in open_wins[:8]:
            try:
                wname = self.kdotool.get_window_name(wid)
                wclass = self.kdotool.get_window_class(wid)
                windows.append(WindowInfo(
                    window_id=wid,
                    name=wname,
                    wm_class=wclass,
                    is_active=(wid == active_id),
                ))
            except Exception:
                pass

        # Extract focused accessibility elements (non-blocking)
        focused_texts: List[str] = []
        summary = ""
        try:
            if self.atspi and self.atspi.enabled:
                focused_texts = self.atspi.get_active_app_text(max_elements=8)
                summary = " | ".join(focused_texts[:5]) if focused_texts else ""
        except Exception as e:
            logger.debug("Failed to extract AT-SPI2 text: %s", e)

        obs = DesktopObservation(
            timestamp=time.time(),
            active_window_id=active_id,
            active_window_name=active_name,
            active_window_class=active_class,
            desktop_number=desktop_num,
            open_window_ids=open_wins,
            active_window_title=active_name,
            all_windows=windows,
            focused_ui_elements=focused_texts,
            ui_summary=summary,
        )

        logger.debug(
            "Observation captured: Active=[%s] '%s' (class: %s) | Open count: %d",
            active_id,
            active_name,
            active_class,
            len(open_wins),
        )
        return obs

