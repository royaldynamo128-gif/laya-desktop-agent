"""
Screen Analyzer: unified UI element perception for Laya Desktop Agent.

Locates UI elements (buttons, labels, menu items, options, fields) using a
robust fallback hierarchy, never hardcoded coordinates:
  Method 1: AT-SPI2 accessibility tree (exact + fuzzy label match) — structured,
            free, instant.
  Method 2: OCR over a fresh screenshot (pytesseract when available) — works on
            apps without accessibility trees.
  Method 3: Keyboard navigation heuristics (delegated to caller).

Clicks located elements via AT-SPI actions (preferred) or synthesized mouse
events at computed screen coordinates (fallback).
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import difflib
import logging
import os
import re
import shutil
import subprocess

logger = logging.getLogger(__name__)


@dataclass
class ElementLocation:
    label: str
    role: str = ""
    method: str = ""                 # atspi | ocr
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    accessible: Any = None           # AT-SPI object when method == atspi
    confidence: float = 0.0

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


class ScreenAnalyzer:
    """Perception layer for element discovery and activation."""

    def __init__(
        self,
        atspi: Any = None,
        screenshot_fn: Optional[Callable[[str], Optional[str]]] = None,
        ocr_fn: Optional[Callable[[str], List[Tuple[str, int, int, int, int]]]] = None,
    ):
        self.atspi = atspi
        self._screenshot_fn = screenshot_fn
        self._ocr_fn = ocr_fn

    # ------------------------------------------------------------------
    # Element discovery
    # ------------------------------------------------------------------

    RESULT_REFERENCE_RE = re.compile(
        r"^(search result|results?|first result|top result|second result|third result|"
        r"the (first|top|second|third|latest|newest) (result|video|link)|video result|result link)$",
        re.IGNORECASE,
    )

    def _is_result_reference(self, label: str) -> bool:
        return bool(self.RESULT_REFERENCE_RE.match(label.strip().lower()))

    def find_element(self, label: str, timeout_sec: float = 0.0) -> Optional[ElementLocation]:
        """
        Find a UI element matching `label` in the currently focused window.
        Tries AT-SPI first, then OCR. Fuzzy matching handles casing and
        minor transcription differences ("5x" vs "5X", "Proceed" vs "proceed").
        """
        if not label:
            return None

        # Method 0: search-result references ("the first result") resolve to
        # link-role elements in visual reading order — page-agnostic, no
        # hardcoded labels or coordinates.
        if self._is_result_reference(label):
            loc = self._find_result_link()
            if loc:
                return loc

        # Method 1: AT-SPI accessibility tree
        loc = self._find_via_atspi(label)
        if loc:
            return loc

        # Method 2: OCR over screenshot
        loc = self._find_via_ocr(label)
        if loc:
            return loc

        return None

    def _find_result_link(self, query: str = "") -> Optional[ElementLocation]:
        """Locate a search-result link: role=link elements in the active
        window, ranked by (query-token relevance, below-search-field, visual
        order). Nav links like 'Home' rank below links that mention the
        search query."""
        if not self.atspi or not getattr(self.atspi, "enabled", False):
            return None
        finder = getattr(self.atspi, "find_elements_by_role", None)
        if not callable(finder):
            return None
        try:
            links = finder("link", max_results=40)
        except Exception as e:
            logger.debug("role-based link search failed: %s", e)
            return None
        if not links:
            return None

        # Top edge of the page's search/entry field (results appear below it)\n        search_y = 0
        try:
            fields = getattr(self.atspi, "get_editable_fields", lambda *a, **k: [])(max_fields=4)
            if fields:
                search_y = min((f.get("y", 0) + f.get("height", 0) or 0) for f in fields)
        except Exception:
            search_y = 0

        q_tokens = {t for t in re.findall(r"\w+", (query or "").lower()) if len(t) > 1}

        def _rank(link: Dict[str, Any]):
            name = (link.get("name") or "").strip().lower()
            overlap = len(q_tokens & set(re.findall(r"\w+", name))) if q_tokens else 0
            below_search = 1 if (not search_y or link.get("y", 0) >= search_y) else 0
            meaningful = 1 if len(name) >= 8 else 0  # 'Home', 'Shorts', 'Sign in' are short
            return (overlap, below_search, meaningful, -link.get("y", 0), -link.get("x", 0))

        ranked = sorted(links, key=_rank, reverse=True)
        for link in ranked:
            name = (link.get("name") or "").strip()
            if not name:
                continue
            return ElementLocation(
                label=name,
                role=link.get("role", "link"),
                method="atspi",
                x=link.get("x", 0),
                y=link.get("y", 0),
                width=link.get("width", 0),
                height=link.get("height", 0),
                accessible=link.get("accessible"),
                confidence=0.85 if q_tokens else 0.7,
            )
        return None

    def element_visible(self, label: str) -> Tuple[bool, str]:
        loc = self.find_element(label)
        if loc:
            return True, f"'{label}' is visible via {loc.method}"
        # Also check raw window title / AT-SPI text content
        if self.atspi:
            try:
                texts = self.atspi.get_active_app_text(max_elements=60)
                for t in texts:
                    if self._labels_match(label, t):
                        return True, f"'{label}' found in screen text"
            except Exception:
                pass
        return False, f"'{label}' not found on screen"

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    def activate(self, location: ElementLocation) -> bool:
        """Activate (click) a located element through the best mechanism."""
        # 1. AT-SPI action interface
        if location.accessible is not None:
            try:
                n = location.accessible.get_n_actions()
                for i in range(n):
                    name = (location.accessible.get_action_name(i) or "").lower()
                    if name in ("click", "press", "activate", "toggle", "check", "select", "jump"):
                        location.accessible.do_action(i)
                        logger.info("Activated element '%s' via AT-SPI action '%s'", location.label, name)
                        return True
                # Fallback: focus + press
                try:
                    comp = location.accessible.get_component_iface()
                    if comp and comp.grab_focus():
                        return self._press_space()
                except Exception:
                    pass
            except Exception as e:
                logger.debug("AT-SPI activation failed: %s", e)

        # 2. Synthetic mouse click at element center
        cx, cy = location.center
        if cx > 0 and cy > 0:
            return self.click_at(cx, cy)

        return False

    def click_at(self, x: int, y: int) -> bool:
        """Synthesize a mouse click at screen coordinates."""
        # AT-SPI mouse event synthesis (works on Wayland)
        if self.atspi and getattr(self.atspi, "enabled", False):
            try:
                import gi
                gi.require_version("Atspi", "2.0")
                from gi.repository import Atspi
                Atspi.generate_mouse_event(x, y, "b1c")
                logger.info("Clicked at (%d, %d) via AT-SPI mouse synthesis", x, y)
                return True
            except Exception as e:
                logger.debug("AT-SPI mouse click failed: %s", e)

        # ydotool fallback (Wayland root permission required)
        ydotool = shutil.which("ydotool") or os.environ.get("YDOTOOL_BIN")
        if ydotool:
            try:
                subprocess.run(
                    [ydotool, "click", "0xC0"],  # left button click
                    check=False, timeout=2.0,
                )
                return True
            except Exception as e:
                logger.debug("ydotool click failed: %s", e)
        return False

    # ------------------------------------------------------------------
    # AT-SPI discovery
    # ------------------------------------------------------------------

    def _find_via_atspi(self, label: str) -> Optional[ElementLocation]:
        if not self.atspi or not getattr(self.atspi, "enabled", False):
            return None
        try:
            elements = self.atspi.find_elements_by_label(label, max_results=8)
        except Exception as e:
            logger.debug("AT-SPI element search failed: %s", e)
            return None

        if not elements:
            return None

        best = elements[0]  # provider sorts by match quality
        acc = best.get("accessible")
        loc = ElementLocation(
            label=best.get("name", label),
            role=best.get("role", ""),
            method="atspi",
            x=best.get("x", 0),
            y=best.get("y", 0),
            width=best.get("width", 0),
            height=best.get("height", 0),
            accessible=acc,
            confidence=best.get("confidence", 0.9),
        )
        logger.info(
            "AT-SPI located '%s' (role=%s, conf=%.2f) at (%d,%d %dx%d)",
            loc.label, loc.role, loc.confidence, loc.x, loc.y, loc.width, loc.height,
        )
        return loc

    # ------------------------------------------------------------------
    # OCR discovery
    # ------------------------------------------------------------------

    def _find_via_ocr(self, label: str) -> Optional[ElementLocation]:
        if self._ocr_fn is None or self._screenshot_fn is None:
            return None
        try:
            shot_path = self._screenshot_fn()
            if not shot_path:
                return None
            words = self._ocr_fn(shot_path)  # [(text, x, y, w, h), ...]
            if not words:
                return None

            # Match single words first, then adjacent word pairs
            candidates: List[ElementLocation] = []
            for i, (text, x, y, w, h) in enumerate(words):
                if self._labels_match(label, text):
                    candidates.append(
                        ElementLocation(label=text, method="ocr", x=x, y=y, width=w, height=h, confidence=0.8)
                    )
                    continue
                # Pair-wise match for multi-word labels ("Proceed" + "button")
                if i + 1 < len(words):
                    t2, x2, y2, w2, h2 = words[i + 1]
                    if self._labels_match(label, f"{text} {t2}") and abs(y - y2) < 12:
                        candidates.append(
                            ElementLocation(
                                label=f"{text} {t2}", method="ocr",
                                x=min(x, x2), y=min(y, y2),
                                width=max(x + w, x2 + w2) - min(x, x2),
                                height=max(y + h, y2 + h2) - min(y, y2),
                                confidence=0.75,
                            )
                        )
            if candidates:
                best = max(candidates, key=lambda c: c.confidence)
                logger.info("OCR located '%s' at (%d,%d)", best.label, best.x, best.y)
                return best
        except Exception as e:
            logger.debug("OCR element search failed: %s", e)
        return None

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    @staticmethod
    def _labels_match(spoken: str, visible: str) -> bool:
        s = spoken.strip().lower()
        v = visible.strip().lower()
        if not s or not v:
            return False
        if s == v:
            return True
        # Substring containment (e.g. spoken "proceed" vs visible "Proceed →")
        if s in v or v in s:
            return True
        # Fuzzy ratio for transcription noise
        ratio = difflib.SequenceMatcher(None, s, v).ratio()
        if ratio >= 0.82:
            return True
        # Token subset: "click 5x" vs visible "5X Upscale"
        s_tokens = set(re.findall(r"\w+", s))
        v_tokens = set(re.findall(r"\w+", v))
        if s_tokens and s_tokens.issubset(v_tokens):
            return True
        return False

    def _press_space(self) -> bool:
        try:
            import gi
            gi.require_version("Atspi", "2.0")
            from gi.repository import Atspi
            Atspi.generate_keyboard_event(0, "space", Atspi.KeySynthType.SYM)
            return True
        except Exception:
            return False
