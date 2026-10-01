"""
Method Priority and Fallback Registry for Laya Desktop Agent.
Enforces the 7-tier action selection hierarchy:
  1. Native application/API/tool integration
  2. Linux desktop accessibility interfaces (AT-SPI2)
  3. KDE/Wayland-compatible automation (kdotool, qdbus6)
  4. Keyboard shortcuts (ydotool, xdotool, key sequences)
  5. Mouse/coordinate interaction (ydotool mouse, click coordinates)
  6. Screen/vision recognition
  7. OCR as a fallback
"""

from enum import IntEnum
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ActionMethod(IntEnum):
    NATIVE_API = 1        # Desktop files, gio, gtk-launch, flatpak run, dbus API, xdg-open
    ACCESSIBILITY = 2     # AT-SPI2 actions, button activation, accessible text insertion
    KDE_AUTOMATION = 3    # kdotool (raise, activate, windowclose, search), KWin scripts, qdbus6
    KEYBOARD = 4          # Global shortcuts, key combos (Alt+Tab, Ctrl+W, Enter, typing)
    MOUSE = 5             # Coordinate mouse click, hover, drag
    VISION = 6            # Screen capture + visual element localization
    OCR = 7               # OCR text detection and coordinate clicking


# Default fallback chains tailored per tool / action category
DEFAULT_METHOD_CHAINS: Dict[str, List[ActionMethod]] = {
    "open_application": [
        ActionMethod.NATIVE_API,       # AppCatalog gtk-launch / gio launch / flatpak
        ActionMethod.ACCESSIBILITY,    # Shell launcher / runner via AT-SPI
        ActionMethod.KDE_AUTOMATION,   # kdotool / KWin activate if already open
        ActionMethod.KEYBOARD,         # App runner shortcut (Alt+Space or Meta) + type
    ],
    "close_application": [
        ActionMethod.KDE_AUTOMATION,   # kdotool windowclose
        ActionMethod.KEYBOARD,         # Alt+F4 / Ctrl+Q / Ctrl+W
        ActionMethod.ACCESSIBILITY,    # Window close button via AT-SPI2
        ActionMethod.NATIVE_API,       # kill / pkill / DBus Stop
    ],
    "switch_window": [
        ActionMethod.KDE_AUTOMATION,   # kdotool windowactivate
        ActionMethod.KEYBOARD,         # Alt+Tab cycling
        ActionMethod.ACCESSIBILITY,    # Window taskbar item via AT-SPI2
        ActionMethod.NATIVE_API,       # KWin DBus activateWindow
    ],
    "type_text": [
        ActionMethod.KEYBOARD,         # ydotool / xdotool type
        ActionMethod.ACCESSIBILITY,    # AT-SPI2 text insertion / editable text interface
        ActionMethod.NATIVE_API,       # Clipboard paste (wl-copy + Ctrl+V)
    ],
    "press_key": [
        ActionMethod.KEYBOARD,         # ydotool / xdotool key
        ActionMethod.ACCESSIBILITY,    # AT-SPI2 generate_keyboard_event
    ],
    "click": [
        ActionMethod.ACCESSIBILITY,    # AT-SPI2 accessible action (Click / Press)
        ActionMethod.MOUSE,            # ydotool mouse click at coordinates
        ActionMethod.KEYBOARD,         # Space / Enter / Tab navigation
        ActionMethod.VISION,           # Vision-based button detection + click
        ActionMethod.OCR,              # OCR button text matching + click
    ],
    "scroll": [
        ActionMethod.KEYBOARD,         # PageUp / PageDown / Up / Down keys
        ActionMethod.MOUSE,            # ydotool mouse scroll wheel
        ActionMethod.ACCESSIBILITY,    # AT-SPI2 Scroll interface
    ],
    "browser_navigate": [
        ActionMethod.NATIVE_API,       # xdg-open URL
        ActionMethod.KEYBOARD,         # Ctrl+L (focus URL bar) + type URL + Enter
        ActionMethod.ACCESSIBILITY,    # AT-SPI2 address bar focus + set text
    ],
    "system_command": [
        ActionMethod.NATIVE_API,       # Dedicated CLI/system tool / DBus
        ActionMethod.KDE_AUTOMATION,   # KWin / Plasma DBus
        ActionMethod.KEYBOARD,         # System keyboard shortcut
    ],
}


class MethodRegistry:
    """Provides prioritized method resolution and method-switching logic."""

    def __init__(self, custom_chains: Optional[Dict[str, List[ActionMethod]]] = None):
        self._chains: Dict[str, List[ActionMethod]] = dict(DEFAULT_METHOD_CHAINS)
        if custom_chains:
            self._chains.update(custom_chains)

    def get_methods_for_action(self, action_name: str) -> List[ActionMethod]:
        """Return the prioritized list of methods for an action."""
        if action_name in self._chains:
            return list(self._chains[action_name])
        # Default generic hierarchy
        return [
            ActionMethod.NATIVE_API,
            ActionMethod.ACCESSIBILITY,
            ActionMethod.KDE_AUTOMATION,
            ActionMethod.KEYBOARD,
            ActionMethod.MOUSE,
            ActionMethod.VISION,
            ActionMethod.OCR,
        ]

    def get_initial_method(self, action_name: str) -> ActionMethod:
        """Get the highest priority method for this action."""
        methods = self.get_methods_for_action(action_name)
        return methods[0] if methods else ActionMethod.NATIVE_API

    def get_next_method(
        self, action_name: str, failed_methods: List[ActionMethod]
    ) -> Optional[ActionMethod]:
        """
        Return the next untried method in the priority chain.
        Returns None if all methods have been exhausted.
        """
        methods = self.get_methods_for_action(action_name)
        for m in methods:
            if m not in failed_methods:
                logger.info(
                    "Method switch for '%s': switching from %s to %s",
                    action_name,
                    [f.name for f in failed_methods],
                    m.name,
                )
                return m
        logger.warning("All methods exhausted for action '%s': %s", action_name, [m.name for m in methods])
        return None
