"""
AT-SPI2 Client for Laya Desktop Agent.
Inspects the Linux desktop accessibility tree to extract structured text, button labels,
and UI state without taking screenshots or using GPU OCR.
"""

from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    import gi
    gi.require_version("Atspi", "2.0")
    from gi.repository import Atspi
    HAS_ATSPI = True
except Exception as e:
    HAS_ATSPI = False
    logger.warning("AT-SPI2 not available: %s", e)


class AtspiClient:
    def __init__(self):
        self.enabled = HAS_ATSPI

    def get_desktop(self) -> Optional[Any]:
        if not self.enabled:
            return None
        try:
            return Atspi.get_desktop(0)
        except Exception as e:
            logger.debug("Failed to get AT-SPI desktop: %s", e)
            return None

    def get_active_app_text(self, max_elements: int = 50) -> List[str]:
        """Extract text from the currently focused application's accessibility tree."""
        if not self.enabled:
            return []

        texts: List[str] = []
        try:
            desktop = self.get_desktop()
            if not desktop:
                return []

            # Find active / focused application
            app_count = desktop.get_child_count()
            for i in range(app_count):
                app = desktop.get_child_at_index(i)
                if not app:
                    continue

                # Check if app has focused window or children
                child_count = app.get_child_count()
                for j in range(child_count):
                    win = app.get_child_at_index(j)
                    if not win:
                        continue
                    
                    state_set = win.get_state_set()
                    if state_set.contains(Atspi.StateType.ACTIVE) or state_set.contains(Atspi.StateType.FOCUSED):
                        # Extract recursive text from this window
                        self._extract_text_recursive(win, texts, max_elements)
                        if texts:
                            return texts

            # If no active window found, fallback to first non-empty window
            for i in range(app_count):
                app = desktop.get_child_at_index(i)
                if app:
                    self._extract_text_recursive(app, texts, max_elements)
                    if texts:
                        break

        except Exception as e:
            logger.debug("Error traversing AT-SPI tree: %s", e)

        return texts

    def _extract_text_recursive(self, accessible: Any, results: List[str], max_elements: int) -> None:
        if len(results) >= max_elements or accessible is None:
            return

        try:
            name = accessible.get_name()
            desc = accessible.get_description()
            
            # Check text interface
            text_iface = accessible.get_text_iface()
            if text_iface:
                char_count = text_iface.get_character_count()
                if char_count > 0:
                    t = text_iface.get_text(0, min(char_count, 1000))
                    if t and t.strip() and t.strip() not in results:
                        results.append(t.strip())

            if name and name.strip() and name.strip() not in results:
                results.append(name.strip())

            child_count = accessible.get_child_count()
            for i in range(child_count):
                if len(results) >= max_elements:
                    break
                child = accessible.get_child_at_index(i)
                if child:
                    self._extract_text_recursive(child, results, max_elements)

        except Exception:
            pass
