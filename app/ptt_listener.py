"""
Global Push-to-Talk (PTT) and Emergency Stop Listener for Laya Desktop Agent.
Uses Linux evdev (/dev/input/event*) to capture global key events on Wayland with zero latency.
"""

from typing import Callable, Optional, List
import logging
import os
import threading
import evdev
from evdev import InputDevice, categorize, ecodes

logger = logging.getLogger(__name__)


class PTTListener:
    def __init__(
        self,
        ptt_key_name: str = "KEY_F12",
        estop_key_name: str = "KEY_F11",
        on_press: Optional[Callable[[], None]] = None,
        on_release: Optional[Callable[[], None]] = None,
        on_estop: Optional[Callable[[], None]] = None,
    ):
        self.ptt_key_code = getattr(ecodes, ptt_key_name, ecodes.KEY_F12)
        self.estop_key_code = getattr(ecodes, estop_key_name, ecodes.KEY_F11)
        self.on_press = on_press
        self.on_release = on_release
        self.on_estop = on_estop

        self._running = False
        self._threads: List[threading.Thread] = []
        self._devices: List[InputDevice] = []
        self._is_pressed = False

    def find_keyboard_devices(self) -> List[InputDevice]:
        """Discover all evdev keyboard input devices."""
        keyboards = []
        try:
            device_paths = [evdev.InputDevice(path) for path in evdev.list_devices()]
            for dev in device_paths:
                caps = dev.capabilities()
                if ecodes.EV_KEY in caps:
                    keys = caps[ecodes.EV_KEY]
                    # Check if device has keyboard-like keys (e.g. KEY_A, KEY_F12)
                    if ecodes.KEY_A in keys or self.ptt_key_code in keys:
                        keyboards.append(dev)
                        logger.debug("Discovered input keyboard device: %s (%s)", dev.name, dev.path)
        except Exception as e:
            logger.warning("Could not list evdev devices: %s", e)

        return keyboards

    def start(self) -> bool:
        """Start listening for PTT and Emergency Stop hotkeys on all keyboard devices."""
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
            "PTT Listener active on %d input device(s). PTT: %s | Emergency Stop: %s",
            len(self._devices),
            evdev.ecodes.KEY[self.ptt_key_code],
            evdev.ecodes.KEY[self.estop_key_code],
        )
        return True

    def _device_loop(self, device: InputDevice) -> None:
        try:
            for event in device.read_loop():
                if not self._running:
                    break

                if event.type == ecodes.EV_KEY:
                    # Emergency Stop Check (Instantaneous latch)
                    if event.code == self.estop_key_code and event.value == 1:
                        logger.critical("EMERGENCY STOP key pressed on %s!", device.name)
                        if self.on_estop:
                            self.on_estop()

                    # Push-to-Talk Check
                    elif event.code == self.ptt_key_code:
                        if event.value == 1 and not self._is_pressed:
                            self._is_pressed = True
                            logger.debug("PTT key pressed.")
                            if self.on_press:
                                self.on_press()
                        elif event.value == 0 and self._is_pressed:
                            self._is_pressed = False
                            logger.debug("PTT key released.")
                            if self.on_release:
                                self.on_release()

        except Exception as e:
            if self._running:
                logger.debug("Device read loop ended for %s: %s", device.name, e)

    def stop(self) -> None:
        self._running = False
        for dev in self._devices:
            try:
                dev.close()
            except Exception:
                pass
        self._devices.clear()
        self._threads.clear()
