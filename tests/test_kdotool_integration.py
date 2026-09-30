"""
Integration Tests for KDE Plasma 6 Wayland kdotool Client.
"""

from integrations.kdotool_client import KdotoolClient


def test_kdotool_active_window_and_desktop():
    kdotool = KdotoolClient()
    active_win = kdotool.get_active_window()
    assert active_win is not None
    assert "{" in active_win and "}" in active_win

    win_name = kdotool.get_window_name(active_win)
    assert isinstance(win_name, str)

    win_class = kdotool.get_window_class(active_win)
    assert isinstance(win_class, str)

    desktop = kdotool.get_desktop()
    assert isinstance(desktop, int)
    assert desktop >= 1

    open_wins = kdotool.search("", limit=5)
    assert len(open_wins) > 0
