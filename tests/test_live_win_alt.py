"""
Live Hardware Injection Tests for Win + Alt Global Shortcut.
Uses Linux kernel evdev.UInput (/dev/uinput) to emit real hardware scancodes
into /dev/input/event* and verifies end-to-end PTTListener detection,
hybrid mode (hold vs tap), right alt, and emergency stop handling.
"""

from typing import List
import time
from unittest.mock import MagicMock
import evdev
from evdev import UInput, ecodes
import pytest
from app.ptt_listener import PTTListener
from core.state import StateManager, AgentState


def _uinput_available() -> bool:
    """Check whether /dev/uinput is usable in this environment."""
    try:
        import evdev
        from evdev import UInput, ecodes

        cap = {
            ecodes.EV_KEY: [
                ecodes.KEY_A,
                ecodes.KEY_LEFTMETA,
                ecodes.KEY_RIGHTALT,
                ecodes.KEY_F12,
            ]
        }
        ui = UInput(cap)
        ui.close()
        return True
    except Exception:
        return False


# These tests inject REAL hardware scancodes; they require /dev/uinput access
# (typically root or an appropriate uinput group). They skip cleanly otherwise
# and run for real on the target machine.
pytestmark = pytest.mark.skipif(
    not _uinput_available(),
    reason="requires /dev/uinput hardware injection access (run on target machine)",
)


@pytest.fixture(scope="module")
def virtual_keyboard():
    """Create a virtual hardware keyboard at the kernel evdev level."""
    cap = {
        ecodes.EV_KEY: [
            ecodes.KEY_A,
            ecodes.KEY_LEFTMETA,
            ecodes.KEY_RIGHTMETA,
            ecodes.KEY_LEFTALT,
            ecodes.KEY_RIGHTALT,
            ecodes.KEY_F12,
            ecodes.KEY_F11,
        ]
    }
    ui = UInput(cap, name="LayaTestVirtualKeyboard")
    time.sleep(0.3)  # Give kernel time to register device node
    yield ui
    ui.close()


def test_live_win_alt_push_to_talk(virtual_keyboard):
    """Test physical Win + Alt held down (> 350ms) as Push-to-Talk."""
    mock_press = MagicMock()
    mock_release = MagicMock()

    listener = PTTListener(
        shortcut="WIN+ALT",
        activation_mode="hybrid",
        hold_threshold_sec=0.25,
        on_press=mock_press,
        on_release=mock_release,
    )
    started = listener.start()
    assert started is True
    time.sleep(0.2)

    try:
        # 1. Press Win alone -> should not activate
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_LEFTMETA, 1)
        virtual_keyboard.syn()
        time.sleep(0.08)
        assert mock_press.call_count == 0
        assert listener._is_active is False

        # 2. Press Alt while Win is down -> Win + Alt satisfied!
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_LEFTALT, 1)
        virtual_keyboard.syn()
        time.sleep(0.12)
        assert mock_press.call_count == 1
        assert listener._is_active is True

        # 3. Hold past threshold (> 0.25s)
        time.sleep(0.30)

        # 4. Release Alt -> combo broken! Should trigger on_release
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_LEFTALT, 0)
        virtual_keyboard.syn()
        time.sleep(0.12)
        assert mock_release.call_count == 1
        assert listener._is_active is False

        # Release Win
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_LEFTMETA, 0)
        virtual_keyboard.syn()
        time.sleep(0.05)

    finally:
        listener.stop()


def test_live_win_alt_toggle_mode(virtual_keyboard):
    """Test physical Win + Alt quick tap (< 250ms) as Toggle Mode."""
    mock_press = MagicMock()
    mock_release = MagicMock()

    listener = PTTListener(
        shortcut="WIN+ALT",
        activation_mode="hybrid",
        hold_threshold_sec=0.25,
        on_press=mock_press,
        on_release=mock_release,
    )
    started = listener.start()
    assert started is True
    time.sleep(0.2)

    try:
        # 1. Quick tap: Press Win + Alt and release within 0.05s
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_LEFTMETA, 1)
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_LEFTALT, 1)
        virtual_keyboard.syn()
        time.sleep(0.05)
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_LEFTALT, 0)
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_LEFTMETA, 0)
        virtual_keyboard.syn()
        time.sleep(0.12)

        # In hybrid mode, quick tap should remain active (toggle listening mode)
        assert mock_press.call_count == 1
        assert mock_release.call_count == 0
        assert listener._is_active is True

        # 2. Second tap while active: Should toggle off and trigger release
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_LEFTMETA, 1)
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_LEFTALT, 1)
        virtual_keyboard.syn()
        time.sleep(0.05)
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_LEFTALT, 0)
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_LEFTMETA, 0)
        virtual_keyboard.syn()
        time.sleep(0.12)

        assert mock_release.call_count == 1
        assert listener._is_active is False

    finally:
        listener.stop()


def test_live_right_alt_combo(virtual_keyboard):
    """Test physical Right Alt + Left Meta."""
    mock_press = MagicMock()
    mock_release = MagicMock()

    listener = PTTListener(
        shortcut="WIN+ALT",
        activation_mode="push_to_talk",
        on_press=mock_press,
        on_release=mock_release,
    )
    started = listener.start()
    assert started is True
    time.sleep(0.2)

    try:
        # Press Right Alt + Left Meta
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_RIGHTALT, 1)
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_LEFTMETA, 1)
        virtual_keyboard.syn()
        time.sleep(0.12)
        assert mock_press.call_count == 1
        assert listener._is_active is True

        # Release Right Alt
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_RIGHTALT, 0)
        virtual_keyboard.syn()
        time.sleep(0.12)
        assert mock_release.call_count == 1
        assert listener._is_active is False

        # Release Left Meta
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_LEFTMETA, 0)
        virtual_keyboard.syn()
        time.sleep(0.05)

    finally:
        listener.stop()


def test_live_emergency_stop_priority(virtual_keyboard):
    """Test physical F11 instant abort priority while Win + Alt is active."""
    mock_press = MagicMock()
    mock_estop = MagicMock()

    listener = PTTListener(
        shortcut="WIN+ALT",
        estop_key_name="KEY_F11",
        on_press=mock_press,
        on_estop=mock_estop,
    )
    started = listener.start()
    assert started is True
    time.sleep(0.2)

    try:
        # Activate Win + Alt
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_LEFTMETA, 1)
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_LEFTALT, 1)
        virtual_keyboard.syn()
        time.sleep(0.12)
        assert listener._is_active is True

        # Hit F11 Emergency Stop
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_F11, 1)
        virtual_keyboard.syn()
        time.sleep(0.12)

        assert mock_estop.call_count == 1
        assert listener._is_active is False

        # Release all keys
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_F11, 0)
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_LEFTALT, 0)
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_LEFTMETA, 0)
        virtual_keyboard.syn()
        time.sleep(0.05)

    finally:
        listener.stop()


def test_live_daemon_state_integration(virtual_keyboard):
    """Test live StateManager state transitions via Win + Alt kernel injection."""
    state_mgr = StateManager()
    assert state_mgr.current_state == AgentState.IDLE

    def on_press():
        state_mgr.set_state(AgentState.LISTENING, reason="Win+Alt down")

    def on_release():
        state_mgr.set_state(AgentState.PROCESSING, reason="Win+Alt up")

    listener = PTTListener(
        shortcut="WIN+ALT",
        activation_mode="push_to_talk",
        on_press=on_press,
        on_release=on_release,
    )
    listener.start()
    time.sleep(0.2)

    try:
        # 1. Press Win + Alt
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_LEFTMETA, 1)
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_LEFTALT, 1)
        virtual_keyboard.syn()
        time.sleep(0.15)
        assert state_mgr.current_state == AgentState.LISTENING

        # 2. Release Win + Alt
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_LEFTALT, 0)
        virtual_keyboard.write(ecodes.EV_KEY, ecodes.KEY_LEFTMETA, 0)
        virtual_keyboard.syn()
        time.sleep(0.15)
        assert state_mgr.current_state == AgentState.PROCESSING

    finally:
        listener.stop()
