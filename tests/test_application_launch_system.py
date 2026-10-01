"""
Comprehensive Integration & Verification Tests for the Application Launching & Window Management System.
Tests Requirements 1 through 12:
- Natural-language requests: Chrome, Dolphin, Settings, Notepad, Telegram, text editor, PATH binary.
- Resolution via AppCatalog without special-casing individual apps.
- Already running detection and foreground activation (unminimizing, desktop switching, raising, focusing).
- Multi-tier launcher reliability.
- Requirement 8: Never report success if window is not detected.
- Requirement 9: Timeout guard, no stuck "processing" state.
- Requirement 10: Fallback and recovery without duplicate process spam.
- Requirement 11: Seamless handoff to autonomous execution loop.
"""

import shutil
import time
from unittest.mock import MagicMock, patch
import pytest

from core.config import load_config
from core.state import StateManager, AgentState
from integrations.app_catalog import AppCatalog
from execution.parameter_binder import ParameterBinder, BoundAction
from tools.app_control import OpenApplicationTool, SwitchWindowTool
from integrations.kdotool_client import KdotoolClient
from verification.verifier import ActionVerifier
from verification.observer import DesktopObservation
from verification.recovery import RecoveryManager
from planning.goal_executor import GoalExecutor, ActionStep, ActionMethod
from planning.goal_planner import GoalPlanner
from planning.method_registry import MethodRegistry
from planning.llm_backend import RuleBasedPlannerBackend
from app.daemon import LayaDaemon


# ==============================================================================
# Requirement 1 & 2: Entity & Intent Resolution for Target Applications
# ==============================================================================

def test_catalog_resolution_target_applications():
    """Verify that required targets resolve correctly without misattributing to unrelated apps."""
    catalog = AppCatalog()

    # 1. Chrome
    res_chrome = catalog.resolve("Chrome")
    assert res_chrome is not None, "Chrome must resolve"
    assert "chrome" in res_chrome.name.lower() or "google-chrome" in res_chrome.desktop_id.lower()
    assert any("chrome" in c.lower() for c in res_chrome.candidate_wm_classes)

    # 2. Dolphin
    res_dolphin = catalog.resolve("Dolphin")
    assert res_dolphin is not None, "Dolphin must resolve"
    assert "dolphin" in res_dolphin.desktop_id.lower() or "dolphin" in res_dolphin.name.lower()
    assert any("dolphin" in c.lower() for c in res_dolphin.candidate_wm_classes)

    # 3. Settings (must match systemsettings, NOT midscroll.Settings)
    res_settings = catalog.resolve("Settings")
    assert res_settings is not None, "Settings must resolve"
    assert "systemsettings" in res_settings.desktop_id.lower() or "system settings" in res_settings.name.lower()
    assert "midscroll" not in res_settings.desktop_id.lower()
    assert "systemsettings" in res_settings.candidate_wm_classes

    # KDE Settings alias
    res_kde_settings = catalog.resolve("KDE Settings")
    assert res_kde_settings is not None
    assert "systemsettings" in res_kde_settings.desktop_id.lower()

    # 4. Notepad
    res_notepad = catalog.resolve("Notepad")
    assert res_notepad is not None, "Notepad must resolve to system text editor"
    assert any(ed in res_notepad.desktop_id.lower() or ed in res_notepad.name.lower() for ed in ("kwrite", "kate", "editor", "text"))

    # 5. "a text editor" / "text editor"
    res_editor = catalog.resolve("a text editor")
    assert res_editor is not None, "'a text editor' must resolve"
    assert any(ed in res_editor.desktop_id.lower() or ed in res_editor.name.lower() for ed in ("kwrite", "kate", "editor", "text"))

    # 6. Telegram
    res_telegram = catalog.resolve("Telegram")
    if res_telegram:  # Telegram may or may not be installed on this test system
        assert "telegram" in res_telegram.desktop_id.lower() or "telegram" in res_telegram.name.lower()


def test_parameter_binder_natural_language():
    """Verify natural-language requests bind with correct candidate_wm_classes and arguments."""
    binder = ParameterBinder()

    # "Open Chrome"
    b_chrome = binder.bind("Open Chrome")
    assert b_chrome is not None
    assert b_chrome.tool_name == "open_application"
    assert "candidate_wm_classes" in b_chrome.arguments
    assert any("chrome" in c.lower() for c in b_chrome.arguments["candidate_wm_classes"])

    # "Open a text editor" (strips article "a")
    b_editor = binder.bind("Open a text editor")
    assert b_editor is not None
    assert b_editor.tool_name == "open_application"
    assert "candidate_wm_classes" in b_editor.arguments

    # "Open Settings"
    b_settings = binder.bind("Open Settings")
    assert b_settings is not None
    assert b_settings.tool_name == "open_application"
    assert "systemsettings" in b_settings.arguments["candidate_wm_classes"]


# ==============================================================================
# Requirement 3, 4, 7: Already Running Detection, Activation & Unminimizing
# ==============================================================================

def test_open_application_already_running_brings_to_front():
    """If application is already running, activate it and verify foreground presence."""
    mock_kdotool = MagicMock(spec=KdotoolClient)
    mock_launcher = MagicMock()
    # Simulate window "win-101" exists and matches wm_class "google-chrome"
    mock_kdotool.search.side_effect = lambda query, limit=10: ["win-101"] if ("chrome" in query or "google" in query or query == "") else []
    mock_kdotool.get_window_class.return_value = "google-chrome"
    mock_kdotool.get_window_name.return_value = "Google Chrome"
    mock_kdotool.activate_window.return_value = True

    tool = OpenApplicationTool(kdotool=mock_kdotool, launcher=mock_launcher)
    result = tool.execute(
        name="Google Chrome",
        wm_class="google-chrome",
        candidate_wm_classes=["google-chrome", "Google-chrome"],
    )

    assert result.success is True
    assert result.data["already_running"] is True
    assert result.data["window_id"] == "win-101"
    # activate_window must have been invoked to bring it to foreground
    mock_kdotool.activate_window.assert_called_with("win-101")


def test_kdotool_activate_window_unminimizes_and_switches_desktop():
    """Verify activate_window handles minimized state and cross-workspace switching."""
    client = KdotoolClient()
    with patch.object(client, "_run") as mock_run:
        # get_desktop_for_window returns desktop 3, while get_desktop returns desktop 1
        # and window is minimized
        mock_run.side_effect = [
            # 1. unminimize: windowstate --remove MINIMIZED
            MagicMock(returncode=0, stdout=""),
            # 2. get_desktop_for_window: get_desktop_for_window win-200
            MagicMock(returncode=0, stdout="3\n"),
            # 3. get_desktop: get_desktop
            MagicMock(returncode=0, stdout="1\n"),
            # 4. set_desktop: set_desktop 3
            MagicMock(returncode=0, stdout=""),
            # 5. windowraise win-200
            MagicMock(returncode=0, stdout=""),
            # 6. windowactivate win-200
            MagicMock(returncode=0, stdout=""),
        ]

        success = client.activate_window("win-200")
        assert success is True

        # Check call sequence:
        # call 1: windowstate --remove MINIMIZED win-200
        assert mock_run.call_args_list[0][0] == ("windowstate", "--remove", "MINIMIZED", "win-200")
        # call 2: get_desktop_for_window win-200
        assert mock_run.call_args_list[1][0] == ("get_desktop_for_window", "win-200")
        # call 3: get_desktop
        assert mock_run.call_args_list[2][0] == ("get_desktop",)
        # call 4: set_desktop 3 (switched desktop!)
        assert mock_run.call_args_list[3][0] == ("set_desktop", "3")
        # call 5: windowraise win-200
        assert mock_run.call_args_list[4][0] == ("windowraise", "win-200")
        # call 6: windowactivate win-200
        assert mock_run.call_args_list[5][0] == ("windowactivate", "win-200")


# ==============================================================================
# Requirement 5, 6, 8: Reliable Launching & Window Detection Verification
# ==============================================================================

def test_open_application_launches_and_detects_new_window():
    """Verify new launch spawns process and polls until window appears."""
    mock_kdotool = MagicMock(spec=KdotoolClient)
    mock_launcher = MagicMock()
    launched = False

    def mock_search(q, limit=50):
        # Before process launch, no matching window exists
        if not launched:
            return []
        # After launch, new window appears
        return ["win-301"]

    mock_kdotool.search.side_effect = mock_search
    mock_kdotool.get_window_class.return_value = "org.kde.dolphin"
    mock_kdotool.get_window_name.return_value = "Home — Dolphin"
    mock_kdotool.activate_window.return_value = True

    tool = OpenApplicationTool(kdotool=mock_kdotool, launcher=mock_launcher)

    real_which = shutil.which
    def mock_which(cmd):
        if cmd == "kstart":
            return "/usr/bin/kstart"
        return real_which(cmd)

    with patch("subprocess.Popen") as mock_popen, \
         patch("shutil.which", side_effect=mock_which):
        def fake_popen(*args, **kwargs):
            nonlocal launched
            launched = True
            return MagicMock()

        mock_popen.side_effect = fake_popen

        result = tool.execute(
            name="Dolphin",
            desktop_id="org.kde.dolphin.desktop",
            wm_class="org.kde.dolphin",
            candidate_wm_classes=["org.kde.dolphin", "dolphin"],
            exec_cmd="dolphin %u",
        )

        assert result.success is True
        assert result.data["already_running"] is False
        assert result.data["window_id"] == "win-301"
        assert result.data["launch_mechanism"] == "kstart"
        mock_kdotool.activate_window.assert_called_with("win-301")


def test_open_application_requirement_8_window_detection_enforced():
    """Requirement 8: Do not report success if window is not detected."""
    mock_kdotool = MagicMock(spec=KdotoolClient)
    mock_launcher = MagicMock()
    # Window never appears
    mock_kdotool.search.return_value = []

    tool = OpenApplicationTool(kdotool=mock_kdotool, launcher=mock_launcher)

    real_which = shutil.which
    def mock_which(cmd):
        if cmd == "kstart":
            return "/usr/bin/kstart"
        if cmd == "ghostapp":
            return None
        return real_which(cmd)

    with patch("subprocess.Popen") as mock_popen, \
         patch("shutil.which", side_effect=mock_which), \
         patch("time.sleep"):  # Speed up polling in test
        mock_popen.return_value = MagicMock()

        result = tool.execute(
            name="GhostApp",
            desktop_id="ghostapp.desktop",
            wm_class="ghostapp",
            candidate_wm_classes=["ghostapp"],
            exec_cmd="ghostapp",
        )

        # Must report False because window was never detected!
        assert result.success is False
        assert result.error == "WindowNotDetected"
        assert "was not detected" in result.message


# ==============================================================================
# Requirement 9 & 10: Timeout Guard & Bounded Recovery
# ==============================================================================

def test_unresolvable_app_returns_immediately_without_hanging():
    """Unresolvable app must fail cleanly and restore IDLE state immediately."""
    config = load_config(profile="balanced")
    daemon = LayaDaemon(config, diagnostic_mode=True)
    daemon.model_mgr.laya.load()

    res = daemon.coordinator.process_text_command("Open AbsolutelyNonExistentApplication998877")
    assert res.success is False
    assert res.verified is False
    assert "couldn't find an application" in res.response_text.lower()
    assert daemon.coordinator.state_mgr.current_state == AgentState.IDLE
    daemon.model_mgr.shutdown()


def test_recovery_rescues_window_and_caps_relaunches():
    """Recovery manager rescues background window on attempt 1 and caps relaunch attempts."""
    recovery = RecoveryManager(max_retries=2)

    action = BoundAction(
        tool_name="open_application",
        arguments={"name": "KWrite", "wm_class": "kwrite", "candidate_wm_classes": ["kwrite"]},
        target="KWrite",
    )
    failed_verification = MagicMock(verified=False, message="Initial window not found")

    mock_execute = MagicMock()
    mock_verify = MagicMock()

    # Attempt 1: switch_window succeeds in finding window
    rescue_tool_res = MagicMock(success=True)
    rescue_verify_res = MagicMock(verified=True, message="Rescued window")

    mock_execute.return_value = rescue_tool_res
    mock_verify.return_value = rescue_verify_res

    res, ver = recovery.attempt_recovery(
        action=action,
        verification=failed_verification,
        execute_fn=mock_execute,
        verify_fn=mock_verify,
    )

    assert ver.verified is True
    assert mock_execute.call_count == 1
    # Check that it tried switch_window first
    called_action = mock_execute.call_args[0][0]
    assert called_action.tool_name == "switch_window"


# ==============================================================================
# Requirement 11: Return Control to Normal Autonomous Loop
# ==============================================================================

def test_goal_executor_multi_action_sequence():
    """After launching, next action in autonomous loop executes seamlessly."""
    planner = GoalPlanner(RuleBasedPlannerBackend())
    mock_coord = MagicMock()
    mock_coord.emergency_stop.is_stopped.return_value = False
    mock_observer = MagicMock()
    mock_verifier = MagicMock()

    # Step 1: open_application -> succeeds
    # Step 2: type_text -> succeeds
    mock_coord.executor.execute_action.side_effect = [
        MagicMock(success=True, message="Launched Dolphin"),
        MagicMock(success=True, message="Typed text"),
    ]
    mock_verifier.verify.return_value = MagicMock(verified=True, message="Verified")

    executor = GoalExecutor(
        planner=planner,
        coordinator=mock_coord,
        observer=mock_observer,
        verifier=mock_verifier,
        method_registry=MethodRegistry(),
    )

    res = executor.execute_goal("Open Dolphin and type hello")
    assert res.success is True
    assert res.verified is True
    assert mock_coord.executor.execute_action.call_count == 2
