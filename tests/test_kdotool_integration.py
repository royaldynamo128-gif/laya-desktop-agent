"""
Integration Tests for KDE Plasma 6 Wayland kdotool Client.
Runs against the hermetic simulated window manager (tests/sim).
"""

import json
from pathlib import Path

from integrations.kdotool_client import KdotoolClient

STATE_PATH = Path(__file__).parent / "sim" / "state" / "wm_state.json"


def _seed_window(name: str = "Seeded Window", wm_class: str = "seeded-app", wid: str = None) -> str:
    wid = wid or "{11111111-2222-3333-4444-555555555555}"
    state = {"windows": {}, "active": None, "desktop": 1, "log": []}
    if STATE_PATH.exists():
        try:
            state = json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    state["windows"][wid] = {"name": name, "class": wm_class, "desktop": 1, "states": [], "pid": 1}
    state["active"] = wid
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state))
    return wid


def test_kdotool_active_window_and_desktop():
    wid = _seed_window("Test Window", "test-class")
    kdotool = KdotoolClient()
    active_win = kdotool.get_active_window()
    assert active_win is not None
    assert "{" in active_win and "}" in active_win

    win_name = kdotool.get_window_name(active_win)
    assert isinstance(win_name, str)
    assert win_name == "Test Window"

    win_class = kdotool.get_window_class(active_win)
    assert isinstance(win_class, str)
    assert win_class == "test-class"

    desktop = kdotool.get_desktop()
    assert isinstance(desktop, int)
    assert desktop >= 1

    open_wins = kdotool.search("", limit=5)
    assert len(open_wins) > 0

    # Search by name and class
    found = kdotool.search("test", limit=5)
    assert wid in found

    # Activate + close roundtrip
    other = _seed_window("Other Window", "other-class", "{99999999-8888-7777-6666-555555555555}")
    assert kdotool.activate_window(other)
    cur = kdotool.get_active_window()
    assert cur == other

    assert kdotool.close_window(other)
    state = json.loads(STATE_PATH.read_text())
    assert other not in state["windows"]


def test_kdotool_window_minimize_state():
    wid = _seed_window("MinimizeMe", "min-class")
    kdotool = KdotoolClient()
    assert kdotool.minimize_window(wid)
    state = json.loads(STATE_PATH.read_text())
    assert "MINIMIZED" in state["windows"][wid]["states"]
    assert state["active"] != wid

    # Unminimize via windowstate --remove
    import subprocess
    subprocess.run([str(Path(__file__).parent / "sim" / "bin" / "kdotool"), "windowstate", "--remove", "MINIMIZED", wid], check=True)
    state = json.loads(STATE_PATH.read_text())
    assert "MINIMIZED" not in state["windows"][wid]["states"]


def test_kdotool_desktop_switching():
    kdotool = KdotoolClient()
    assert kdotool.set_desktop(3)
    assert kdotool.get_desktop() == 3
    kdotool.set_desktop(1)
