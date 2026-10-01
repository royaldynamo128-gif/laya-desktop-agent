"""
Unit and integration tests for natural desktop commands and unobtrusive parameter binding.
"""

import pytest
from execution.parameter_binder import ParameterBinder
from models.laya_engine import LayaDecision
from verification.verifier import ActionVerifier
from verification.observer import DesktopObservation
from tools.base import ToolResult


def make_decision(intent="other", confidence=0.9):
    return LayaDecision(
        intent=intent,
        confidence=confidence,
        safety_level="safe",
        requires_confirmation=False,
        requires_observation=True,
        raw_decision={},
        latency_ms=10.0,
    )


def test_natural_open_commands():
    binder = ParameterBinder()
    
    for phrase in ["Open Chrome", "Launch Chrome", "Start Chrome", "open chrome"]:
        action = binder.bind(phrase, make_decision("other"))
        assert action is not None, f"Failed for {phrase}"
        assert action.tool_name == "open_application"
        assert "chrome" in action.arguments.get("name", "").lower()

    for phrase in ["Open Telegram", "Launch Telegram", "open telegram"]:
        action = binder.bind(phrase, make_decision("other"))
        assert action is not None
        assert action.tool_name == "open_application"
        assert "telegram" in action.arguments.get("name", "").lower()

    for phrase in ["Open terminal", "Launch terminal"]:
        action = binder.bind(phrase, make_decision("other"))
        assert action is not None
        assert action.tool_name == "open_application"
        assert any(t in action.arguments.get("name", "").lower() for t in ["terminal", "ghostty", "konsole"])


def test_natural_close_commands():
    binder = ParameterBinder()

    for phrase in ["Close Chrome", "Quit Chrome", "exit chrome"]:
        action = binder.bind(phrase, make_decision("other"))
        assert action is not None
        assert action.tool_name == "close_application"
        assert "chrome" in action.arguments.get("name", "").lower()

    for phrase in ["Close this", "Close this window", "close it"]:
        action = binder.bind(phrase, make_decision("other"))
        assert action is not None
        assert action.tool_name == "close_application"
        assert "name" not in action.arguments or not action.arguments["name"]


def test_natural_switch_commands():
    binder = ParameterBinder()

    for phrase in ["Switch to Telegram", "Focus Telegram", "Bring up Telegram", "Go to Telegram"]:
        action = binder.bind(phrase, make_decision("other"))
        assert action is not None
        assert action.tool_name == "switch_window"
        assert "telegram" in action.arguments.get("query", "").lower()

    for phrase in ["Switch to Chrome", "Go to Chrome"]:
        action = binder.bind(phrase, make_decision("other"))
        assert action is not None
        assert action.tool_name == "switch_window"
        assert "chrome" in action.arguments.get("query", "").lower()


def test_natural_input_and_window_commands():
    binder = ParameterBinder()

    action = binder.bind("Type hello world", make_decision("other"))
    assert action is not None
    assert action.tool_name == "type_text"
    assert action.arguments.get("text") == "hello world"

    action = binder.bind("Press Enter", make_decision("other"))
    assert action is not None
    assert action.tool_name == "press_key"
    assert action.arguments.get("key") == "Return"

    action = binder.bind("Scroll down", make_decision("other"))
    assert action is not None
    assert action.tool_name == "scroll"
    assert action.arguments.get("direction") == "down"

    action = binder.bind("Scroll up", make_decision("other"))
    assert action is not None
    assert action.tool_name == "scroll"
    assert action.arguments.get("direction") == "up"

    action = binder.bind("Take a screenshot", make_decision("other"))
    assert action is not None
    assert action.tool_name == "take_screenshot"

    action = binder.bind("Minimize this window", make_decision("other"))
    assert action is not None
    assert action.tool_name == "minimize_window"

    action = binder.bind("Maximize this window", make_decision("other"))
    assert action is not None
    assert action.tool_name == "maximize_window"


def test_switch_window_verification_previous():
    verifier = ActionVerifier()
    binder = ParameterBinder()
    action = binder.bind("Switch to previous", make_decision("other"))
    
    import time
    now = time.time()
    pre = DesktopObservation(timestamp=now, active_window_id="win-1", active_window_name="App 1", active_window_class="app1", desktop_number=1, open_window_ids=["win-1", "win-2"])
    post = DesktopObservation(timestamp=now, active_window_id="win-2", active_window_name="App 2", active_window_class="app2", desktop_number=1, open_window_ids=["win-1", "win-2"])
    res = verifier.verify(action, ToolResult(success=True, message="Switched"), pre, post)
    assert res.verified is True
