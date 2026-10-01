"""
Unit tests for Goal Planner, decomposition, and RuleBasedPlannerBackend.
"""

import pytest
from planning.goal_planner import GoalPlanner
from planning.llm_backend import RuleBasedPlannerBackend, AutoPlannerBackend
from planning.execution_memory import ExecutionMemory, ActionStep
from planning.method_registry import ActionMethod
from tools.base import ToolResult
from verification.observer import DesktopObservation
from verification.verifier import VerificationResult



def test_rule_based_planner_single_action():
    backend = RuleBasedPlannerBackend()
    plan = backend.plan_goal("Open Google Chrome", {})
    assert plan.is_single_action is True
    assert len(plan.steps) == 1
    assert plan.steps[0].intent_hint == "open_application"
    assert "chrome" in plan.steps[0].target.lower()


def test_rule_based_planner_compound_goal():
    backend = RuleBasedPlannerBackend()
    goal = "Open Firefox, go to example.com, and search for Linux"
    plan = backend.plan_goal(goal, {})
    assert plan.is_single_action is False
    assert len(plan.steps) == 3
    assert plan.steps[0].intent_hint == "open_application"
    assert "firefox" in plan.steps[0].target.lower()
    assert plan.steps[1].intent_hint == "browser_navigate"
    assert "example.com" in plan.steps[1].target
    assert plan.steps[2].intent_hint in ("browser_search", "browser_navigate")
    assert "linux" in plan.steps[2].target.lower()


def test_rule_based_planner_open_and_type():
    backend = RuleBasedPlannerBackend()
    goal = "Open Notepad and write hello world"
    plan = backend.plan_goal(goal, {})
    assert len(plan.steps) == 2
    assert plan.steps[0].intent_hint == "open_application"
    assert "notepad" in plan.steps[0].target.lower()
    assert plan.steps[1].intent_hint == "type_text"
    assert plan.steps[1].target == "hello world"


def test_goal_planner_replan():
    planner = GoalPlanner(AutoPlannerBackend())
    goal = "Open Dolphin and take a screenshot"
    obs = DesktopObservation(
        timestamp=0.0,
        active_window_id="win-1",
        active_window_name="Dolphin",
        active_window_class="dolphin",
        desktop_number=1,
        open_window_ids=["win-1"],
    )
    plan = planner.plan(goal, obs)
    memory = ExecutionMemory(goal, plan)

    # Simulate step 1 was completed
    step1 = plan.steps[0]
    t_res = ToolResult(success=True, message="Done")
    v_res = VerificationResult(verified=True, message="Verified", details={})
    memory.record_success(step1, "open_application", ActionMethod.NATIVE_API, t_res, v_res, 50.0)

    # Replan when step 2 had an issue
    replan = planner.replan(goal, memory, failed_step=plan.steps[1], desktop_state=obs)
    # The new plan should not repeat step 1
    for s in replan.steps:
        sig = f"{s.intent_hint.lower()}:{s.target.lower()}"
        assert sig != f"{step1.intent_hint.lower()}:{step1.target.lower()}"



def test_is_goal_complete_check():
    planner = GoalPlanner(AutoPlannerBackend())
    goal = "Open Dolphin"
    plan = planner.plan(goal)
    memory = ExecutionMemory(goal, plan)

    # Incomplete initially
    complete, msg = planner.is_goal_complete(goal, memory)
    assert complete is False

    # Mark step complete
    memory.current_step_index = 1
    memory.completed_steps.append(
        type("CompletedMock", (), {"step": plan.steps[0], "tool_used": "open_application"})()
    )
    complete, msg = planner.is_goal_complete(goal, memory)
    assert complete is True
