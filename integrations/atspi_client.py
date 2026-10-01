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

    def get_focused_element_text(self) -> str:
        """Find the currently focused UI element and return its text/name."""
        if not self.enabled:
            return ""
        try:
            desktop = self.get_desktop()
            if not desktop:
                return ""
            for i in range(desktop.get_child_count()):
                app = desktop.get_child_at_index(i)
                if not app:
                    continue
                for j in range(app.get_child_count()):
                    win = app.get_child_at_index(j)
                    if not win:
                        continue
                    elem = self._find_focused_child(win)
                    if elem:
                        txt_iface = elem.get_text_iface()
                        if txt_iface and txt_iface.get_character_count() > 0:
                            return txt_iface.get_text(0, 500).strip()
                        name = elem.get_name()
                        if name:
                            return name.strip()
        except Exception as e:
            logger.debug("Error getting focused element: %s", e)
        return ""

    def _find_focused_child(self, accessible: Any) -> Optional[Any]:
        try:
            state = accessible.get_state_set()
            if state.contains(Atspi.StateType.FOCUSED):
                return accessible
            for i in range(accessible.get_child_count()):
                child = accessible.get_child_at_index(i)
                if child:
                    found = self._find_focused_child(child)
                    if found:
                        return found
        except Exception:
            pass
        return None

    def get_visible_buttons(self, max_buttons: int = 15) -> List[str]:
        """Discover clickable button labels in the currently active window."""
        if not self.enabled:
            return []
        buttons: List[str] = []
        try:
            desktop = self.get_desktop()
            if not desktop:
                return []
            for i in range(desktop.get_child_count()):
                app = desktop.get_child_at_index(i)
                if not app:
                    continue
                for j in range(app.get_child_count()):
                    win = app.get_child_at_index(j)
                    if not win:
                        continue
                    state = win.get_state_set()
                    if state.contains(Atspi.StateType.ACTIVE) or state.contains(Atspi.StateType.FOCUSED):
                        self._collect_buttons_recursive(win, buttons, max_buttons)
                        if buttons:
                            return buttons
        except Exception as e:
            logger.debug("Error discovering buttons: %s", e)
        return buttons

    def _collect_buttons_recursive(self, accessible: Any, results: List[str], max_buttons: int) -> None:
        if len(results) >= max_buttons or accessible is None:
            return
        try:
            role_name = accessible.get_role_name()
            if role_name in ("push button", "button", "toggle button"):
                name = accessible.get_name()
                if name and name.strip() and name.strip() not in results:
                    results.append(name.strip())
            for i in range(accessible.get_child_count()):
                if len(results) >= max_buttons:
                    break
                child = accessible.get_child_at_index(i)
                if child:
                    self._collect_buttons_recursive(child, results, max_buttons)
        except Exception:
            pass

    def get_window_summary(self) -> str:
        """Return a compact textual summary of visible UI text in active window."""
        texts = self.get_active_app_text(max_elements=10)
        return " | ".join(texts) if texts else ""

    # ------------------------------------------------------------------
    # Element discovery for UI automation (click / select / activate)
    # ------------------------------------------------------------------

    CLICKABLE_ROLES = (
        "push button", "button", "toggle button", "check box", "checkbox",
        "radio button", "menu item", "list item", "tree item", "tab",
        "page tab", "link", "label", "option", "combo box", "menu",
    )

    def find_elements_by_label(self, label: str, max_results: int = 8) -> List[Dict[str, Any]]:
        """
        Find UI elements matching `label` in the ACTIVE window.
        Returns list of dicts sorted by match confidence:
          {name, role, x, y, width, height, accessible, confidence}
        Exact matches rank above fuzzy ones; interactive roles rank above
        passive text.
        """
        if not self.enabled or not label:
            return []

        import difflib
        import re as _re

        target = label.strip().lower()
        results: List[Dict[str, Any]] = []

        try:
            desktop = self.get_desktop()
            if not desktop:
                return []

            for win in self._active_windows(desktop):
                self._collect_matching_elements(win, target, results, max_results * 3)

        except Exception as e:
            logger.debug("find_elements_by_label failed: %s", e)

        # Rank: interactive role > exact match > fuzzy ratio
        def _rank(r: Dict[str, Any]) -> float:
            score = r["confidence"]
            if r.get("role", "") in self.CLICKABLE_ROLES:
                score += 0.15
            return score

        results.sort(key=_rank, reverse=True)
        return results[:max_results]

    def find_elements_by_role(self, role: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Find UI elements by accessibility ROLE in the ACTIVE window
        (e.g. role="link" for search-result links). Returned in tree order;
        callers sort by screen position as needed.
        """
        if not self.enabled or not role:
            return []
        results: List[Dict[str, Any]] = []
        try:
            desktop = self.get_desktop()
            if not desktop:
                return []
            for win in self._active_windows(desktop):
                self._collect_role_elements(win, role.lower(), results, max_results)
        except Exception as e:
            logger.debug("find_elements_by_role failed: %s", e)
        return results[:max_results]

    def _active_windows(self, desktop: Any) -> List[Any]:
        """Windows of the active/foreground application (fallback: first)."""
        active_windows: List[Any] = []
        for i in range(desktop.get_child_count()):
            app = desktop.get_child_at_index(i)
            if not app:
                continue
            for j in range(app.get_child_count()):
                win = app.get_child_at_index(j)
                if not win:
                    continue
                try:
                    state = win.get_state_set()
                    if state.contains(Atspi.StateType.ACTIVE) or state.contains(Atspi.StateType.FOCUSED):
                        active_windows.append(win)
                except Exception:
                    continue
        if not active_windows:
            # Fall back to first app's first window
            for i in range(desktop.get_child_count()):
                app = desktop.get_child_at_index(i)
                if app and app.get_child_count() > 0:
                    active_windows.append(app.get_child_at_index(0))
                    break
        return active_windows

    def _collect_role_elements(
        self, accessible: Any, role: str, results: List[Dict[str, Any]], limit: int, depth: int = 0,
    ) -> None:
        if len(results) >= limit or accessible is None or depth > 25:
            return
        try:
            node_role = (accessible.get_role_name() or "").lower()
            if node_role == role:
                name = (accessible.get_name() or "").strip()
                x = y = w = h = 0
                try:
                    ext = accessible.get_extents()
                    if ext:
                        x, y, w, h = ext.x, ext.y, ext.width, ext.height
                except Exception:
                    pass
                results.append({
                    "name": name, "role": node_role,
                    "x": x, "y": y, "width": w, "height": h,
                    "accessible": accessible,
                    "confidence": 1.0,
                })
            for i in range(accessible.get_child_count()):
                if len(results) >= limit:
                    return
                child = accessible.get_child_at_index(i)
                if child:
                    self._collect_role_elements(child, role, results, limit, depth + 1)
        except Exception:
            pass

    def _collect_matching_elements(
        self, accessible: Any, target: str, results: List[Dict[str, Any]], limit: int, depth: int = 0,
    ) -> None:
        if len(results) >= limit or accessible is None or depth > 25:
            return
        try:
            name = (accessible.get_name() or "").strip()
            role = accessible.get_role_name() or ""

            if name:
                v = name.lower()
                exact = v == target
                substr = target in v or v in target
                ratio = difflib.SequenceMatcher(None, target, v).ratio()
                # Token subset match: "5x" in "5X Upscale option"
                t_tokens = set(_re.findall(r"\w+", target))
                v_tokens = set(_re.findall(r"\w+", v))
                subset = bool(t_tokens) and t_tokens.issubset(v_tokens)

                if exact or substr or ratio >= 0.82 or subset:
                    conf = 1.0 if exact else (0.9 if subset or substr else ratio)
                    # Element extents via Component interface
                    x = y = w = h = 0
                    try:
                        ext = accessible.get_extents()
                        if ext:
                            x, y, w, h = ext.x, ext.y, ext.width, ext.height
                    except Exception:
                        pass
                    results.append({
                        "name": name, "role": role,
                        "x": x, "y": y, "width": w, "height": h,
                        "accessible": accessible,
                        "confidence": round(conf, 3),
                    })

            for i in range(accessible.get_child_count()):
                if len(results) >= limit:
                    return
                child = accessible.get_child_at_index(i)
                if child:
                    self._collect_matching_elements(child, target, results, limit, depth + 1)
        except Exception:
            pass

    def get_editable_fields(self, max_fields: int = 10) -> List[Dict[str, Any]]:
        """Find text-entry fields in the active window (for form filling)."""
        if not self.enabled:
            return []
        fields: List[Dict[str, Any]] = []
        try:
            desktop = self.get_desktop()
            if not desktop:
                return []
            win = None
            for i in range(desktop.get_child_count()):
                app = desktop.get_child_at_index(i)
                if not app:
                    continue
                for j in range(app.get_child_count()):
                    w = app.get_child_at_index(j)
                    if not w:
                        continue
                    try:
                        state = w.get_state_set()
                        if state.contains(Atspi.StateType.ACTIVE) or state.contains(Atspi.StateType.FOCUSED):
                            win = w
                            break
                    except Exception:
                        continue
                if win:
                    break
            if win:
                self._collect_editable(win, fields, max_fields)
        except Exception as e:
            logger.debug("get_editable_fields failed: %s", e)
        return fields

    def _collect_editable(self, accessible: Any, fields: List[Dict[str, Any]], limit: int, depth: int = 0) -> None:
        if len(fields) >= limit or accessible is None or depth > 25:
            return
        try:
            role = accessible.get_role_name() or ""
            name = (accessible.get_name() or "").strip()
            if role in ("text", "entry", "password text", "search", "text box", "document text", "edit bar"):
                try:
                    state = accessible.get_state_set()
                    editable = state.contains(Atspi.StateType.EDITABLE)
                except Exception:
                    editable = True
                if editable:
                    x = y = w2 = h2 = 0
                    try:
                        ext = accessible.get_extents()
                        if ext:
                            x, y, w2, h2 = ext.x, ext.y, ext.width, ext.height
                    except Exception:
                        pass
                    fields.append({
                        "name": name, "role": role,
                        "x": x, "y": y, "width": w2, "height": h2,
                        "accessible": accessible,
                    })
            for i in range(accessible.get_child_count()):
                if len(fields) >= limit:
                    return
                child = accessible.get_child_at_index(i)
                if child:
                    self._collect_editable(child, fields, limit, depth + 1)
        except Exception:
            pass

