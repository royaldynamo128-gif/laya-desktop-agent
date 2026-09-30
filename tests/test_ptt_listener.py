"""
Unit Tests for PTTListener Multi-Key Global Shortcut and Activation Modes.
"""

import time
from unittest.mock import MagicMock
from evdev import ecodes
from app.ptt_listener import PTTListener


def test_shortcut_parsing():
    listener = PTTListener(shortcut="WIN+ALT")
    assert len(listener._combo_groups) == 2
    # First group has left/right meta
    assert ecodes.KEY_LEFTMETA in listener._combo_groups[0]
    assert ecodes.KEY_RIGHTMETA in listener._combo_groups[0]
    # Second group has left/right alt
    assert ecodes.KEY_LEFTALT in listener._combo_groups[1]
    assert ecodes.KEY_RIGHTALT in listener._combo_groups[1]

    # Meta+Alt alias
    listener_meta = PTTListener(shortcut="META+ALT")
    assert len(listener_meta._combo_groups) == 2

    # Single key
    listener_f12 = PTTListener(shortcut="KEY_F12")
    assert len(listener_f12._combo_groups) == 1
    assert ecodes.KEY_F12 in listener_f12._combo_groups[0]


def test_win_alt_combo_push_to_talk():
    mock_press = MagicMock()
    mock_release = MagicMock()

    listener = PTTListener(
        shortcut="WIN+ALT",
        activation_mode="push_to_talk",
        on_press=mock_press,
        on_release=mock_release,
    )

    # 1. Press Win alone -> should not activate
    listener._handle_key_event(ecodes.KEY_LEFTMETA, 1)
    time.sleep(0.02)
    mock_press.assert_not_called()
    assert listener._is_active is False

    # 2. Press Alt while Win is held -> combo satisfied! Should activate
    listener._handle_key_event(ecodes.KEY_LEFTALT, 1)
    time.sleep(0.02)
    mock_press.assert_called_once()
    assert listener._is_active is True

    # 3. Release Alt -> combo broken! Should release immediately in PTT mode
    listener._handle_key_event(ecodes.KEY_LEFTALT, 0)
    time.sleep(0.02)
    mock_release.assert_called_once()
    assert listener._is_active is False


def test_right_alt_combo():
    mock_press = MagicMock()
    mock_release = MagicMock()

    listener = PTTListener(
        shortcut="WIN+ALT",
        activation_mode="push_to_talk",
        on_press=mock_press,
        on_release=mock_release,
    )

    # Press Right Alt + Left Meta
    listener._handle_key_event(ecodes.KEY_RIGHTALT, 1)
    listener._handle_key_event(ecodes.KEY_LEFTMETA, 1)
    time.sleep(0.02)
    mock_press.assert_called_once()
    assert listener._is_active is True

    # Release Left Meta
    listener._handle_key_event(ecodes.KEY_LEFTMETA, 0)
    time.sleep(0.02)
    mock_release.assert_called_once()
    assert listener._is_active is False


def test_hybrid_mode_tap_vs_hold():
    mock_press = MagicMock()
    mock_release = MagicMock()

    listener = PTTListener(
        shortcut="WIN+ALT",
        activation_mode="hybrid",
        hold_threshold_sec=0.20,
        on_press=mock_press,
        on_release=mock_release,
    )

    # A. Tap Test: Press and quickly release (< 0.20s)
    listener._handle_key_event(ecodes.KEY_LEFTMETA, 1)
    listener._handle_key_event(ecodes.KEY_LEFTALT, 1)
    time.sleep(0.05)  # Quick tap
    listener._handle_key_event(ecodes.KEY_LEFTALT, 0)
    listener._handle_key_event(ecodes.KEY_LEFTMETA, 0)
    time.sleep(0.02)

    # Press should have triggered, but release should NOT have triggered yet (staying active in toggle mode)
    assert mock_press.call_count == 1
    assert mock_release.call_count == 0
    assert listener._is_active is True

    # Tap again while active -> should stop recording
    listener._handle_key_event(ecodes.KEY_LEFTMETA, 1)
    listener._handle_key_event(ecodes.KEY_LEFTALT, 1)
    time.sleep(0.02)
    assert mock_release.call_count == 1
    assert listener._is_active is False

    # Clean key state
    listener._handle_key_event(ecodes.KEY_LEFTALT, 0)
    listener._handle_key_event(ecodes.KEY_LEFTMETA, 0)

    # B. Hold Test: Press and hold (> 0.20s)
    mock_press.reset_mock()
    mock_release.reset_mock()

    listener._handle_key_event(ecodes.KEY_LEFTMETA, 1)
    listener._handle_key_event(ecodes.KEY_LEFTALT, 1)
    time.sleep(0.25)  # Hold past threshold
    listener._handle_key_event(ecodes.KEY_LEFTALT, 0)
    time.sleep(0.02)

    # Both press and release should be called
    assert mock_press.call_count == 1
    assert mock_release.call_count == 1
    assert listener._is_active is False


def test_emergency_stop_key_priority():
    mock_estop = MagicMock()
    mock_press = MagicMock()

    listener = PTTListener(
        shortcut="WIN+ALT",
        estop_key_name="KEY_F11",
        on_press=mock_press,
        on_estop=mock_estop,
    )

    # Active combo
    listener._handle_key_event(ecodes.KEY_LEFTMETA, 1)
    listener._handle_key_event(ecodes.KEY_LEFTALT, 1)
    assert listener._is_active is True

    # Hit F11 Emergency Stop
    listener._handle_key_event(ecodes.KEY_F11, 1)
    mock_estop.assert_called_once()
    assert listener._is_active is False
    assert len(listener._pressed_keys) == 0


def test_fallback_f12_key():
    mock_press = MagicMock()
    mock_release = MagicMock()

    listener = PTTListener(
        shortcut="WIN+ALT",
        ptt_key_name="KEY_F12",
        on_press=mock_press,
        on_release=mock_release,
    )

    # Single F12 press
    listener._handle_key_event(ecodes.KEY_F12, 1)
    time.sleep(0.02)
    mock_press.assert_called_once()
    assert listener._is_active is True

    # Single F12 release
    listener._handle_key_event(ecodes.KEY_F12, 0)
    time.sleep(0.02)
    mock_release.assert_called_once()
    assert listener._is_active is False
