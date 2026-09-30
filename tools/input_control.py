"""
Input Control Tools for Laya Desktop Agent.
Handles typing text, pressing keys, and scrolling on KDE Wayland via AT-SPI2 with wtype fallback.
"""

from typing import Optional
import logging
import shutil
import subprocess
import time
from tools.base import Tool, ToolResult, SafetyTier

logger = logging.getLogger(__name__)

try:
    import gi
    gi.require_version("Atspi", "2.0")
    from gi.repository import Atspi
    HAS_ATSPI = True
except Exception as e:
    HAS_ATSPI = False
    logger.debug("AT-SPI2 keyboard/mouse synthesis not available: %s", e)


class ScrollTool(Tool):
    name = "scroll"
    description = "Scroll up or down in the active window (e.g. 'scroll down', 'scroll up twice')"
    safety_tier = SafetyTier.SAFE

    def execute(self, direction: str = "down", count: int = 1, **kwargs) -> ToolResult:
        dir_clean = direction.lower().strip()
        key = "Page_Down" if "down" in dir_clean else "Page_Up"
        mouse_button = "b5c" if "down" in dir_clean else "b4c"
        repeat = max(1, min(count, 10))
        success = True

        if HAS_ATSPI:
            try:
                for _ in range(repeat):
                    # Try Page_Down/Page_Up keysym via AT-SPI
                    ok = Atspi.generate_keyboard_event(0, key, Atspi.KeySynthType.SYM)
                    if not ok:
                        # Fallback to mouse wheel click event
                        ok = Atspi.generate_mouse_event(0, 0, mouse_button)
                    time.sleep(0.08)
                return ToolResult(
                    success=True,
                    message=f"Scrolled {direction} {count} time(s) via AT-SPI.",
                    data={"direction": direction, "count": count, "method": "atspi"},
                    observation_required=False,
                )
            except Exception as e:
                logger.warning("AT-SPI scroll failed (%s), attempting wtype fallback", e)

        # Fallback to wtype
        wtype = shutil.which("wtype") or "/home/rai/.local/bin/wtype"
        for _ in range(repeat):
            try:
                subprocess.run([wtype, "-k", key], check=False, timeout=2.0)
                time.sleep(0.08)
            except Exception as e:
                logger.error("Error scrolling via %s: %s", wtype, e)
                success = False

        return ToolResult(
            success=success,
            message=f"Scrolled {direction} {count} time(s).",
            data={"direction": direction, "count": count, "method": "wtype"},
            error=None if success else "Failed to send scroll keys",
            observation_required=False,
        )


class TypeTextTool(Tool):
    name = "type_text"
    description = "Type text into the currently focused window"
    safety_tier = SafetyTier.SAFE

    def execute(self, text: str, **kwargs) -> ToolResult:
        if not text:
            return ToolResult(success=False, message="Text cannot be empty.", error="Missing text")

        if HAS_ATSPI:
            try:
                ok = Atspi.generate_keyboard_event(0, text, Atspi.KeySynthType.STRING)
                if ok:
                    return ToolResult(
                        success=True,
                        message=f"Typed text: '{text[:20]}...'",
                        data={"length": len(text), "method": "atspi"},
                        observation_required=False,
                    )
            except Exception as e:
                logger.warning("AT-SPI text synthesis failed (%s), trying wtype fallback", e)

        wtype = shutil.which("wtype") or "/home/rai/.local/bin/wtype"
        try:
            subprocess.run([wtype, text], check=False, timeout=5.0)
            return ToolResult(
                success=True,
                message=f"Typed text: '{text[:20]}...'",
                data={"length": len(text), "method": "wtype"},
                observation_required=False,
            )
        except Exception as e:
            return ToolResult(success=False, message="Failed to type text.", error=str(e))


class PressKeyTool(Tool):
    name = "press_key"
    description = "Press a keyboard key (e.g. Return, Enter, Tab, Escape, Space)"
    safety_tier = SafetyTier.SAFE

    KEY_MAP = {
        "enter": "Return",
        "return": "Return",
        "tab": "Tab",
        "escape": "Escape",
        "esc": "Escape",
        "backspace": "BackSpace",
        "space": "space",
        "up": "Up",
        "down": "Down",
        "left": "Left",
        "right": "Right",
        "page_down": "Page_Down",
        "pagedown": "Page_Down",
        "page_up": "Page_Up",
        "pageup": "Page_Up",
        "home": "Home",
        "end": "End",
    }

    def execute(self, key: str, **kwargs) -> ToolResult:
        mapped_key = self.KEY_MAP.get(key.lower().strip(), key)

        if HAS_ATSPI:
            try:
                ok = Atspi.generate_keyboard_event(0, mapped_key, Atspi.KeySynthType.SYM)
                if ok:
                    return ToolResult(
                        success=True,
                        message=f"Pressed key '{mapped_key}'.",
                        data={"key": mapped_key, "method": "atspi"},
                        observation_required=False,
                    )
            except Exception as e:
                logger.warning("AT-SPI key press failed (%s), trying wtype fallback", e)

        wtype = shutil.which("wtype") or "/home/rai/.local/bin/wtype"
        try:
            subprocess.run([wtype, "-k", mapped_key], check=False, timeout=2.0)
            return ToolResult(
                success=True,
                message=f"Pressed key '{mapped_key}'.",
                data={"key": mapped_key, "method": "wtype"},
                observation_required=False,
            )
        except Exception as e:
            return ToolResult(success=False, message=f"Failed to press key {key}.", error=str(e))
