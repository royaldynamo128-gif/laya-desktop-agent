"""
Unit tests for GoalExecutor autonomous loop and method fallback.
"""

from unittest.mock import MagicMock
import pytest
from planning.goal_executor import GoalExecutor, GoalResult
from planning.goal_planner import GoalPlanner
from planning.llm_backend import RuleBasedPlannerBackend
from planning.method_registry import ActionMethod, MethodRegistry
from tools.base import ToolResult
from verification.observer import DesktopObservation
from verification.verifier import VerificationResult


def test_goal_executor_single_action_fast_path():
    planner = GoalPlanner(RuleBasedPlannerBackend())
    coordinator = MagicMock()
    pipe_res = MagicMock(success=True, verified=True, response_text="Chrome opened")
    coordinator.process_text_command.return_value = pipe_res

    observer = MagicMock()
    observer.observe.return_value = DesktopObservation(
        timestamp=0.0,
        active_window_id="1",
        active_window_name="Chrome",
        active_window_class="google-chrome",
        desktop_number=1,
        open_window_ids=["1"],
    )
    verifier = MagicMock()

    executor = GoalExecutor(
        planner=planner,
        coordinator=coordinator,
        observer=observer,
        verifier=verifier,
    )

    res = executor.execute_goal("Open Chrome")
    assert res.success is True
    assert res.verified is True
    assert res.total_steps_executed == 1
    coordinator.process_text_command.assert_called_once()


def test_goal_executor_multi_step_execution():
    planner = GoalPlanner(RuleBasedPlannerBackend())
    coordinator = MagicMock()
    coordinator.emergency_stop.is_stopped.return_value = False
    coordinator.status_broadcaster = None
    coordinator.executor.execute_action.return_value = ToolResult(success=True, message="Executed successfully")

    observer = MagicMock()
    observer.observe.return_value = DesktopObservation(
        timestamp=0.0,
        active_window_id="1",
        active_window_name="Terminal",
        active_window_class="ghostty",
        desktop_number=1,
        open_window_ids=["1"],
    )

    verifier = MagicMock()
    verifier.verify.return_value = VerificationResult(verified=True, message="Verified", details={})

    executor = GoalExecutor(
        planner=planner,
        coordinator=coordinator,
        observer=observer,
        verifier=verifier,
    )

    # Multi-step goal
    res = executor.execute_goal("Open Terminal and type echo hello")
    assert res.success is True
    assert res.verified is True
    assert res.total_steps_executed == 2
    assert coordinator.executor.execute_action.call_count >= 2


def test_goal_executor_emergency_stop_abort():
    planner = GoalPlanner(RuleBasedPlannerBackend())
    coordinator = MagicMock()
    coordinator.emergency_stop.is_stopped.return_value = True  # E-stop latched
    coordinator.status_broadcaster = None

    observer = MagicMock()
    verifier = MagicMock()

    executor = GoalExecutor(
        planner=planner,
        coordinator=coordinator,
        observer=observer,
        verifier=verifier,
    )

    res = executor.execute_goal("Open Chrome and search Linux")
    assert res.success is False
    assert "Emergency Stop" in res.message
