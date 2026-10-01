"""
Input Control Tools for Laya Desktop Agent.
Typing text, pressing keys, and scrolling on KDE Wayland.

Input synthesis hierarchy (never a hardcoded binary path):
  1. AT-SPI2 keyboard/mouse event synthesis (works on Wayland).
  2. wtype (resolved via PATH or $WTYPE_BIN env override).
  3. Clipboard paste for long text (wl-copy + Ctrl+V).

Key combos ("ctrl+w") are supported via wtype modifier flags or AT-SPI
sequential synthesis.
"""

from typing import Optional
import logging
import os
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


def _wtype_path() -> Optional[str]:
    """Resolve wtype without hardcoded user paths."""
    return shutil.which("wtype") or os.environ.get("WTYPE_BIN") or None


# Modifier keysyms AT-SPI understands
_ATSPI_MODIFIERS = {
    "ctrl": "Control_L",
    "control": "Control_L",
    "alt": "Alt_L",
    "shift": "Shift_L",
    "super": "Super_L",
    "meta": "Super_L",
    "win": "Super_L",
}

_WTYPE_MODIFIERS = {
    "ctrl": "ctrl", "control": "ctrl",
    "alt": "alt", "shift": "shift",
    "super": "super", "meta": "super", "win": "super",
}


def _parse_combo(key: str):
    """Parse 'ctrl+w' → (['ctrl'], 'w'). Single keys → ([], key)."""
    parts = [p for p in key.replace("-", "+").split("+") if p]
    if not parts:
        return [], key
    mods = []
    final = parts[-1]
    for p in parts[:-1]:
        p = p.lower().strip()
        if p in _ATSPI_MODIFIERS:
            mods.append(p)
        else:
            # Non-modifier in modifier position: treat whole string as single key
            return [], key
    return mods, final


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
                    ok = Atspi.generate_keyboard_event(0, key, Atspi.KeySynthType.SYM)
                    if not ok:
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

        wtype = _wtype_path()
        if not wtype:
            return ToolResult(
                success=False,
                message="No input synthesis mechanism available (AT-SPI or wtype required).",
                error="NoInputMechanism",
            )
        for _ in range(repeat):
            try:
                subprocess.run([wtype, "-k", key], check=False, timeout=2.0)
                time.sleep(0.08)
            except Exception as e:
                logger.error("Error scrolling via wtype: %s", e)
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

    def execute(self, text: str, method: str = "auto", **kwargs) -> ToolResult:
        if not text:
            return ToolResult(success=False, message="Text cannot be empty.", error="Missing text")

        # Clipboard paste path (for long text or explicit request)
        if method == "clipboard" or (method == "auto" and len(text) > 120):
            ok = self._type_via_clipboard(text)
            if ok:
                return ToolResult(
                    success=True,
                    message=f"Pasted text ({len(text)} chars).",
                    data={"length": len(text), "method": "clipboard"},
                    observation_required=False,
                )

        if HAS_ATSPI and method != "wtype":
            try:
                ok = Atspi.generate_keyboard_event(0, text, Atspi.KeySynthType.STRING)
                if ok:
                    return ToolResult(
                        success=True,
                        message=f"Typed text: '{text[:20]}...'" if len(text) > 20 else f"Typed text: '{text}'",
                        data={"length": len(text), "method": "atspi"},
                        observation_required=False,
                    )
            except Exception as e:
                logger.warning("AT-SPI text synthesis failed (%s), trying wtype fallback", e)

        wtype = _wtype_path()
        if not wtype:
            return ToolResult(
                success=False,
                message="No text input mechanism available (AT-SPI, wtype, or clipboard).",
                error="NoInputMechanism",
            )
        try:
            subprocess.run([wtype, text], check=False, timeout=5.0)
            return ToolResult(
                success=True,
                message=f"Typed text: '{text[:20]}...'" if len(text) > 20 else f"Typed text: '{text}'",
                data={"length": len(text), "method": "wtype"},
                observation_required=False,
            )
        except Exception as e:
            return ToolResult(success=False, message="Failed to type text.", error=str(e))

    def _type_via_clipboard(self, text: str) -> bool:
        """Copy to clipboard then Ctrl+V (fast and reliable for long text)."""
        try:
            wl_copy = shutil.which("wl-copy") or os.environ.get("WL_COPY_BIN")
            if wl_copy:
                subprocess.run(
                    [wl_copy, text], input=text.encode(), check=False, timeout=2.0,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                time.sleep(0.1)
                # Ctrl+V via AT-SPI or wtype
                if HAS_ATSPI:
                    ok = self._combo_via_atspi(["ctrl"], "v")
                    if ok:
                        return True
                wtype = _wtype_path()
                if wtype:
                    subprocess.run(
                        [wtype, "-M", "ctrl", "-k", "v", "-m", "ctrl"],
                        check=False, timeout=2.0,
                    )
                    return True
        except Exception as e:
            logger.debug("Clipboard typing failed: %s", e)
        return False

    @staticmethod
    def _combo_via_atspi(mods, final_key: str) -> bool:
        try:
            for m in mods:
                Atspi.generate_keyboard_event(0, _ATSPI_MODIFIERS[m], Atspi.KeySynthType.SYM)
            Atspi.generate_keyboard_event(0, final_key, Atspi.KeySynthType.SYM)
            for m in reversed(mods):
                mod_up = _ATSPI_MODIFIERS[m]
                # AT-SPI has no key-up: acceptable for most combos via wtype instead
            return True
        except Exception:
            return False


class PressKeyTool(Tool):
    name = "press_key"
    description = "Press a keyboard key or combo (e.g. Return, Escape, ctrl+w)"
    safety_tier = SafetyTier.SAFE

    KEY_MAP = {
        "enter": "Return", "return": "Return",
        "tab": "Tab", "escape": "Escape", "esc": "Escape",
        "backspace": "BackSpace", "space": "space",
        "up": "Up", "down": "Down", "left": "Left", "right": "Right",
        "page_down": "Page_Down", "pagedown": "Page_Down",
        "page_up": "Page_Up", "pageup": "Page_Up",
        "home": "Home", "end": "End",
        "delete": "Delete", "del": "Delete", "insert": "Insert",
        "f1": "F1", "f2": "F2", "f3": "F3", "f4": "F4", "f5": "F5", "f6": "F6",
        "f7": "F7", "f8": "F8", "f9": "F9", "f10": "F10", "f11": "F11", "f12": "F12",
    }

    def execute(self, key: str, method: str = "auto", **kwargs) -> ToolResult:
        if not key:
            return ToolResult(success=False, message="Key required.", error="MissingKey")

        raw = key.strip()
        lower = raw.lower()

        # Normalize single keys through the map; keep combos intact
        if "+" not in raw and "-" not in raw:
            mapped = self.KEY_MAP.get(lower, raw)
            mods, final = [], mapped
        else:
            mods, final = _parse_combo(raw)
            if not mods:
                # e.g. "ctrl-w" where parse failed — normalize the final key
                final = self.KEY_MAP.get(final.lower(), final)

        if mods:
            return self._press_combo(mods, final, method)
        return self._press_single(final, method)

    def _press_single(self, key: str, method: str) -> ToolResult:
        if HAS_ATSPI and method != "wtype":
            try:
                ok = Atspi.generate_keyboard_event(0, key, Atspi.KeySynthType.SYM)
                if ok:
                    return ToolResult(
                        success=True,
                        message=f"Pressed key '{key}'.",
                        data={"key": key, "method": "atspi"},
                        observation_required=False,
                    )
            except Exception as e:
                logger.warning("AT-SPI key press failed (%s), trying wtype fallback", e)

        wtype = _wtype_path()
        if not wtype:
            return ToolResult(
                success=False,
                message="No key input mechanism available.",
                error="NoInputMechanism",
            )
        try:
            subprocess.run([wtype, "-k", key], check=False, timeout=2.0)
            return ToolResult(
                success=True,
                message=f"Pressed key '{key}'.",
                data={"key": key, "method": "wtype"},
                observation_required=False,
            )
        except Exception as e:
            return ToolResult(success=False, message=f"Failed to press key {key}.", error=str(e))

    def _press_combo(self, mods, final: str, method: str) -> ToolResult:
        combo_str = "+".join(mods + [final])

        # wtype handles combos cleanly (press/release modifiers)
        wtype = _wtype_path()
        if wtype:
            args = []
            for m in mods:
                args.extend(["-M", _WTYPE_MODIFIERS.get(m, m)])
            args.extend(["-k", final])
            for m in reversed(mods):
                args.extend(["-m", _WTYPE_MODIFIERS.get(m, m)])
            try:
                subprocess.run([wtype] + args, check=False, timeout=2.0)
                return ToolResult(
                    success=True,
                    message=f"Pressed key combo '{combo_str}'.",
                    data={"key": combo_str, "method": "wtype"},
                    observation_required=False,
                )
            except Exception as e:
                logger.warning("wtype combo failed: %s", e)

        if HAS_ATSPI:
            try:
                for m in mods:
                    Atspi.generate_keyboard_event(0, _ATSPI_MODIFIERS[m], Atspi.KeySynthType.SYM)
                Atspi.generate_keyboard_event(0, final, Atspi.KeySynthType.SYM)
                return ToolResult(
                    success=True,
                    message=f"Pressed key combo '{combo_str}' (AT-SPI).",
                    data={"key": combo_str, "method": "atspi"},
                    observation_required=False,
                )
            except Exception as e:
                logger.warning("AT-SPI combo failed: %s", e)

        return ToolResult(
            success=False,
            message=f"No mechanism available to press combo '{combo_str}'.",
            error="NoInputMechanism",
        )
