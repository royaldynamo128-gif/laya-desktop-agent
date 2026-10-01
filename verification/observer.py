"""
Desktop State Observer for Laya Desktop Agent.

Latency-aware observation:
  - Lightweight observation (kdotool only) with a short TTL cache, so the
    3-4 observations per command/step don't each re-run ~20 subprocesses.
  - Deep observation (AT-SPI UI text) is opt-in: only when a consumer
    actually needs screen text (read_screen, click steps, verification).
  - observe(delay_sec=...) invalidates the cache so post-action checks see
    fresh state.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging
import threading
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
    active_window_title: str = ""
    all_windows: List[WindowInfo] = field(default_factory=list)
    focused_ui_elements: List[str] = field(default_factory=list)
    ui_summary: str = ""

    def __post_init__(self):
        if not self.active_window_title:
            self.active_window_title = self.active_window_name


class DesktopObserver:
    """Cached, lightweight desktop state observer."""

    def __init__(self, kdotool: KdotoolClient, atspi: AtspiClient, cache_ttl_sec: float = 0.35):
        self.kdotool = kdotool
        self.atspi = atspi
        self.cache_ttl_sec = cache_ttl_sec
        self._cache: Optional[DesktopObservation] = None
        self._cache_lock = threading.Lock()
        self._invalidated = False

    def invalidate(self) -> None:
        """Force the next observe() to capture fresh state (call after actions)."""
        with self._cache_lock:
            self._invalidated = True

    def observe(self, delay_sec: float = 0.0) -> DesktopObservation:
        """
        Capture current desktop state. Uses the TTL cache when the state is
        fresh (within cache_ttl_sec) and not invalidated. delay_sec > 0 always
        bypasses the cache (post-action settle time).
        """
        if delay_sec > 0:
            time.sleep(delay_sec)
            with self._cache_lock:
                self._invalidated = True

        with self._cache_lock:
            if (
                self._cache is not None
                and not self._invalidated
                and (time.time() - self._cache.timestamp) < self.cache_ttl_sec
            ):
                return self._cache

        obs = self._capture()
        with self._cache_lock:
            self._invalidated = False
            self._cache = obs
        return obs

    def observe_deep(self, delay_sec: float = 0.0) -> DesktopObservation:
        """Observation including AT-SPI UI text (expensive — use when needed)."""
        obs = self.observe(delay_sec=delay_sec)
        if obs.ui_summary:
            return obs

        focused_texts: List[str] = []
        summary = ""
        try:
            if self.atspi and self.atspi.enabled:
                focused_texts = self.atspi.get_active_app_text(max_elements=8)
                summary = " | ".join(focused_texts[:5]) if focused_texts else ""
        except Exception as e:
            logger.debug("Failed to extract AT-SPI2 text: %s", e)

        obs.focused_ui_elements = focused_texts
        obs.ui_summary = summary
        return obs

    def _capture(self) -> DesktopObservation:
        active_id = self.kdotool.get_active_window()
        active_name = self.kdotool.get_window_name(active_id) if active_id else ""
        active_class = self.kdotool.get_window_class(active_id) if active_id else ""
        desktop_num = self.kdotool.get_desktop()
        open_wins = self.kdotool.search("", limit=15)

        windows: List[WindowInfo] = []
        for wid in open_wins[:8]:
            try:
                wname = self.kdotool.get_window_name(wid)
                wclass = self.kdotool.get_window_class(wid)
                windows.append(
                    WindowInfo(
                        window_id=wid,
                        name=wname,
                        wm_class=wclass,
                        is_active=(wid == active_id),
                    )
                )
            except Exception:
                pass

        obs = DesktopObservation(
            timestamp=time.time(),
            active_window_id=active_id,
            active_window_name=active_name,
            active_window_class=active_class,
            desktop_number=desktop_num,
            open_window_ids=open_wins,
            active_window_title=active_name,
            all_windows=windows,
        )

        logger.debug(
            "Observation: Active=[%s] '%s' (class: %s) | Open count: %d",
            active_id, active_name, active_class, len(open_wins),
        )
        return obs
