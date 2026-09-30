"""
Global Shortcut and Push-to-Talk (PTT) / Emergency Stop Listener for Laya Desktop Agent.
Uses Linux evdev (/dev/input/event*) to capture global key combinations (e.g. WIN+ALT)
and single keys on Wayland with zero latency across all applications.
"""

from typing import Callable, Optional, List, Set
import logging
import threading
import time
import evdev
from evdev import InputDevice, categorize, ecodes

logger = logging.getLogger(__name__)

MODIFIER_MAP = {
    "win": {ecodes.KEY_LEFTMETA, ecodes.KEY_RIGHTMETA},
    "meta": {ecodes.KEY_LEFTMETA, ecodes.KEY_RIGHTMETA},
    "super": {ecodes.KEY_LEFTMETA, ecodes.KEY_RIGHTMETA},
    "alt": {ecodes.KEY_LEFTALT, ecodes.KEY_RIGHTALT},
    "ctrl": {ecodes.KEY_LEFTCTRL, ecodes.KEY_RIGHTCTRL},
    "control": {ecodes.KEY_LEFTCTRL, ecodes.KEY_RIGHTCTRL},
    "shift": {ecodes.KEY_LEFTSHIFT, ecodes.KEY_RIGHTSHIFT},
}


class PTTListener:
    def __init__(
        self,
        shortcut: str = "WIN+ALT",
        ptt_key_name: str = "KEY_F12",
        estop_key_name: str = "KEY_F11",
        activation_mode: str = "hybrid",
        hold_threshold_sec: float = 0.35,
        on_press: Optional[Callable[[], None]] = None,
        on_release: Optional[Callable[[], None]] = None,
        on_estop: Optional[Callable[[], None]] = None,
    ):
        self.shortcut_str = shortcut.strip().upper()
        self.activation_mode = activation_mode.lower().strip()
        self.hold_threshold_sec = hold_threshold_sec
        self.on_press = on_press
        self.on_release = on_release
        self.on_estop = on_estop

        # Parse primary combo groups
        self._combo_groups: List[Set[int]] = self._parse_shortcut(self.shortcut_str)
        
        # Parse fallback and emergency stop keys
        self.fallback_key_code: Optional[int] = getattr(ecodes, ptt_key_name, ecodes.KEY_F12) if ptt_key_name else None
        self.estop_key_code: int = getattr(ecodes, estop_key_name, ecodes.KEY_F11)

        self._running = False
        self._threads: List[threading.Thread] = []
        self._devices: List[InputDevice] = []
        
        # State tracking
        self._lock = threading.Lock()
        self._pressed_keys: Set[int] = set()
        self._combo_down = False
        self._is_active = False  # whether recording is currently active
        self._press_timestamp = 0.0

    def _parse_shortcut(self, shortcut_str: str) -> List[Set[int]]:
        """Parse shortcut like 'WIN+ALT' or 'META+ALT' into required key groups."""
        groups: List[Set[int]] = []
        parts = [p.strip().lower() for p in shortcut_str.split("+") if p.strip()]
        for part in parts:
            if part in MODIFIER_MAP:
                groups.append(set(MODIFIER_MAP[part]))
            else:
                up = part.upper()
                if not up.startswith("KEY_"):
                    up_key = f"KEY_{up}"
                else:
                    up_key = up
                code = getattr(ecodes, up_key, getattr(ecodes, up, None))
                if code is not None:
                    groups.append({code})
                else:
                    logger.warning("Unrecognized key in shortcut '%s': %s", shortcut_str, part)
        return groups

    def find_keyboard_devices(self) -> List[InputDevice]:
        """Discover all evdev keyboard input devices."""
        keyboards = []
        try:
            device_paths = [evdev.InputDevice(path) for path in evdev.list_devices()]
            for dev in device_paths:
                caps = dev.capabilities()
                if ecodes.EV_KEY in caps:
                    keys = caps[ecodes.EV_KEY]
                    # Check if device has standard keyboard keys
                    if ecodes.KEY_A in keys or ecodes.KEY_LEFTMETA in keys or ecodes.KEY_LEFTALT in keys:
                        keyboards.append(dev)
                        logger.debug("Discovered input keyboard device: %s (%s)", dev.name, dev.path)
        except Exception as e:
            logger.warning("Could not list evdev devices: %s", e)

        return keyboards

    def start(self) -> bool:
        """Start listening for shortcut and Emergency Stop hotkeys on all keyboard devices."""
        self._devices = self.find_keyboard_devices()
        if not self._devices:
            logger.warning("No evdev keyboard devices found. Global hotkey listener inactive.")
            return False

        self._running = True

        for dev in self._devices:
            t = threading.Thread(
                target=self._device_loop,
                args=(dev,),
                daemon=True,
                name=f"EvdevListener-{dev.name[:10]}",
            )
            t.start()
            self._threads.append(t)

        logger.info(
            "Global Shortcut Listener active on %d device(s). Shortcut: %s | Mode: %s | Emergency Stop: %s",
            len(self._devices),
            self.shortcut_str,
            self.activation_mode,
            evdev.ecodes.KEY.get(self.estop_key_code, str(self.estop_key_code)),
        )
        return True

    def _device_loop(self, device: InputDevice) -> None:
        try:
            for event in device.read_loop():
                if not self._running:
                    break

                if event.type == ecodes.EV_KEY:
                    self._handle_key_event(event.code, event.value, device.name)

        except Exception as e:
            if self._running:
                logger.debug("Device read loop ended for %s: %s", device.name, e)

    def _handle_key_event(self, code: int, value: int, device_name: str = "") -> None:
        """Central event handler for key press (1), repeat (2), and release (0)."""
        # 1. Emergency Stop Check
        if code == self.estop_key_code and value == 1:
            logger.critical("EMERGENCY STOP key pressed on %s!", device_name or "keyboard")
            with self._lock:
                self._is_active = False
                self._combo_down = False
                self._pressed_keys.clear()
            if self.on_estop:
                self.on_estop()
            return

        with self._lock:
            # Update pressed keys set
            if value in (1, 2):
                self._pressed_keys.add(code)
            elif value == 0:
                self._pressed_keys.discard(code)

            # Check secondary fallback key (e.g. F12)
            if self.fallback_key_code and code == self.fallback_key_code:
                if value == 1:
                    self._activate_press()
                elif value == 0:
                    self._activate_release(held_long=True)
                return

            # Check primary combination (e.g. Win + Alt)
            combo_satisfied = bool(self._combo_groups) and all(
                any(k in self._pressed_keys for k in group)
                for group in self._combo_groups
            )

            # State transition: Combo just became pressed
            if combo_satisfied and not self._combo_down:
                self._combo_down = True
                self._press_timestamp = time.time()

                if not self._is_active:
                    self._is_active = True
                    logger.debug("Shortcut %s activated (mode: %s).", self.shortcut_str, self.activation_mode)
                    if self.on_press:
                        threading.Thread(target=self.on_press, daemon=True).start()
                else:
                    # In toggle or hybrid mode: second tap stops recording
                    if self.activation_mode in ("toggle", "hybrid"):
                        self._is_active = False
                        logger.debug("Shortcut %s tapped while active: stopping recording.", self.shortcut_str)
                        if self.on_release:
                            threading.Thread(target=self.on_release, daemon=True).start()

            # State transition: Combo released
            elif not combo_satisfied and self._combo_down:
                self._combo_down = False
                duration = time.time() - self._press_timestamp

                if self.activation_mode == "push_to_talk":
                    if self._is_active:
                        self._is_active = False
                        logger.debug("Shortcut %s released after %.2fs in PTT mode.", self.shortcut_str, duration)
                        if self.on_release:
                            threading.Thread(target=self.on_release, daemon=True).start()

                elif self.activation_mode == "hybrid":
                    if duration >= self.hold_threshold_sec:
                        # Held long enough to be Push-to-Talk
                        if self._is_active:
                            self._is_active = False
                            logger.debug("Shortcut %s held for %.2fs -> Push-to-Talk release.", self.shortcut_str, duration)
                            if self.on_release:
                                threading.Thread(target=self.on_release, daemon=True).start()
                    else:
                        # Quick tap (< 350ms): Stay in toggle listening mode!
                        logger.debug("Shortcut %s quick tap (%.2fs) -> Keeping toggle active.", self.shortcut_str, duration)

    def _activate_press(self) -> None:
        if not self._is_active:
            self._is_active = True
            logger.debug("Fallback activation key pressed.")
            if self.on_press:
                threading.Thread(target=self.on_press, daemon=True).start()

    def _activate_release(self, held_long: bool = True) -> None:
        if self._is_active:
            self._is_active = False
            logger.debug("Fallback activation key released.")
            if self.on_release:
                threading.Thread(target=self.on_release, daemon=True).start()

    def trigger_activation(self) -> None:
        """Programmatic activation trigger (e.g. from CLI or DBus)."""
        with self._lock:
            if not self._is_active:
                self._is_active = True
                if self.on_press:
                    threading.Thread(target=self.on_press, daemon=True).start()
            else:
                self._is_active = False
                if self.on_release:
                    threading.Thread(target=self.on_release, daemon=True).start()

    def set_inactive(self) -> None:
        """Reset active recording state (e.g. when VAD silence auto-stops)."""
        with self._lock:
            self._is_active = False

    def stop(self) -> None:
        self._running = False
        for dev in self._devices:
            try:
                dev.close()
            except Exception:
                pass
        self._devices.clear()
        self._threads.clear()
