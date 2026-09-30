"""
Unit Tests for Tool Registry and Tool Execution.
"""

from tools.base import Tool, ToolResult, SafetyTier
from tools.registry import ToolRegistry


class EchoTool(Tool):
    name = "echo_tool"
    description = "Echo back parameter"
    safety_tier = SafetyTier.SAFE

    def execute(self, text: str = "default", **kwargs) -> ToolResult:
        return ToolResult(success=True, message=f"Echo: {text}", data={"text": text})


def test_tool_registry_registration_and_execution():
    registry = ToolRegistry()
    assert registry.has_tool("echo_tool") is False

    tool = EchoTool()
    registry.register(tool)
    assert registry.has_tool("echo_tool") is True
    assert registry.get("echo_tool") == tool

    res = tool.execute(text="hello voice agent")
    assert res.success is True
    assert res.message == "Echo: hello voice agent"
    assert res.data["text"] == "hello voice agent"

    tool_list = registry.list_tools()
    assert len(tool_list) == 1
    assert tool_list[0]["name"] == "echo_tool"
    assert tool_list[0]["safety_tier"] == "SAFE"


def test_open_application_folder_detection(monkeypatch):
    from tools.app_control import OpenApplicationTool
    from unittest.mock import MagicMock
    from pathlib import Path

    mock_launcher = MagicMock()
    mock_kdotool = MagicMock()
    mock_kdotool.search.return_value = []

    tool = OpenApplicationTool(launcher=mock_launcher, kdotool=mock_kdotool)

    # 1. Test folder detection logic
    p = tool._detect_folder("my projects folder")
    assert p is not None
    assert p == Path.home() / "Projects"

    dl = tool._detect_folder("downloads folder")
    assert dl is not None
    assert dl == Path.home() / "Downloads"

    # Mock subprocess.Popen to verify it launches file manager
    opened_cmds = []
    monkeypatch.setattr("subprocess.Popen", lambda cmd, **kwargs: opened_cmds.append(cmd))

    res = tool.execute("my projects folder")
    assert res.success is True
    assert res.data["is_folder"] is True
    assert len(opened_cmds) == 1
    assert "Projects" in opened_cmds[0][1]


def test_switch_window_category_mapping():
    from tools.app_control import SwitchWindowTool
    from unittest.mock import MagicMock

    mock_kdotool = MagicMock()
    # Direct search for "browser" returns nothing, but "google-chrome" returns a window ID
    def mock_search(query, limit=1):
        if query == "google-chrome":
            return ["{test-chrome-window-id}"]
        if query == "ghostty":
            return ["{test-ghostty-window-id}"]
        return []

    mock_kdotool.search.side_effect = mock_search
    mock_kdotool.get_window_name.return_value = "Google Chrome"

    tool = SwitchWindowTool(kdotool=mock_kdotool)

    # Test "the browser" mapping
    res = tool.execute(query="the browser")
    assert res.success is True
    assert res.data["window_id"] == "{test-chrome-window-id}"
    mock_kdotool.activate_window.assert_called_with("{test-chrome-window-id}")

    # Test "terminal" mapping
    mock_kdotool.get_window_name.return_value = "Ghostty Terminal"
    res_term = tool.execute(query="terminal")
    assert res_term.success is True
    assert res_term.data["window_id"] == "{test-ghostty-window-id}"


def test_input_control_tools():
    from tools.input_control import TypeTextTool, PressKeyTool, ScrollTool

    type_tool = TypeTextTool()
    res_type = type_tool.execute(text="Test input")
    assert res_type.success is True

    key_tool = PressKeyTool()
    res_key = key_tool.execute(key="enter")
    assert res_key.success is True
    assert res_key.data["key"] == "Return"

    scroll_tool = ScrollTool()
    res_scroll = scroll_tool.execute(direction="down", count=2)
    assert res_scroll.success is True
    assert res_scroll.data["count"] == 2

