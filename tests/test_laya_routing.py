"""
Unit Tests for Laya Decision Routing and Parameter Binding.
"""

from execution.parameter_binder import ParameterBinder
from models.laya_engine import LayaDecision


def test_parameter_binder_mappings():
    binder = ParameterBinder()

    # 1. Open Application
    dec = LayaDecision(
        intent="open_application",
        confidence=0.95,
        safety_level="safe",
        requires_confirmation=False,
        requires_observation=True,
        raw_decision={},
        latency_ms=30.0,
    )
    action = binder.bind("Open Google Chrome", dec)
    assert action is not None
    assert action.tool_name == "open_application"
    assert "chrome" in action.arguments["name"].lower()

    # 2. Close Application
    dec.intent = "close_application"
    action = binder.bind("Close this window", dec)
    assert action is not None
    assert action.tool_name == "close_application"

    # 3. Switch Window
    dec.intent = "switch_window"
    action = binder.bind("Switch to Opera", dec)
    assert action is not None
    assert action.tool_name == "switch_window"
    assert "opera" in action.arguments["query"].lower()

    # 4. Scroll
    dec.intent = "scroll"
    action = binder.bind("Scroll down twice", dec)
    assert action is not None
    assert action.tool_name == "scroll"
    assert action.arguments["direction"] == "down"
    assert action.arguments["count"] == 2

    # 5. Browser Search
    dec.intent = "browser_action"
    action = binder.bind("Search YouTube for Linux tutorials", dec)
    assert action is not None
    assert action.tool_name == "browser_search"
    assert action.arguments["engine"] == "youtube"
    assert "linux" in action.arguments["query"].lower()

    # 6. Screenshot
    dec.intent = "screen_query"
    action = binder.bind("Take a screenshot", dec)
    assert action is not None
    assert action.tool_name == "take_screenshot"

    # 7. Volume
    dec.intent = "system_setting"
    action = binder.bind("Set volume to 50 percent", dec)
    assert action is not None
    assert action.tool_name == "set_volume"
    assert "50" in str(action.arguments["level"])


def test_parameter_binder_folder_and_category_mappings():
    binder = ParameterBinder()

    # 1. Folder opening
    dec = LayaDecision(
        intent="open_application",
        confidence=0.95,
        safety_level="safe",
        requires_confirmation=False,
        requires_observation=True,
        raw_decision={},
        latency_ms=25.0,
    )
    action = binder.bind("Open my Projects folder", dec)
    assert action is not None
    assert action.tool_name == "open_application"
    assert "project" in action.arguments["name"].lower()

    action_dl = binder.bind("Open the downloads folder", dec)
    assert action_dl is not None
    assert action_dl.tool_name == "open_application"
    assert "download" in action_dl.arguments["name"].lower()

    # 2. Category window switching
    dec.intent = "switch_window"
    action_br = binder.bind("Switch to the browser", dec)
    assert action_br is not None
    assert action_br.tool_name == "switch_window"
    assert "browser" in action_br.arguments["query"].lower()

    action_term = binder.bind("Switch to terminal", dec)
    assert action_term is not None
    assert action_term.tool_name == "switch_window"
    assert "terminal" in action_term.arguments["query"].lower()

