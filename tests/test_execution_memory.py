"""
Unit tests for ExecutionMemory and state tracking.
"""

import pytest
from planning.execution_memory import ActionStep, ExecutionMemory, GoalPlan
from planning.method_registry import ActionMethod
from tools.base import ToolResult
from verification.verifier import VerificationResult


def test_execution_memory_tracking():
    step1 = ActionStep(step_id=1, description="Open Chrome", intent_hint="open_application", target="chrome")
    step2 = ActionStep(step_id=2, description="Type test", intent_hint="type_text", target="test")
    plan = GoalPlan(goal_id="g1", original_goal="Open Chrome and type test", steps=[step1, step2])
    memory = ExecutionMemory(original_goal="Open Chrome and type test", plan=plan)

    assert memory.get_current_step() == step1
    assert memory.is_plan_exhausted is False

    # Track method attempt
    count = memory.record_method_attempt(1, ActionMethod.NATIVE_API)
    assert count == 1
    assert memory.get_method_attempt_count(1, ActionMethod.NATIVE_API) == 1

    # Record success of step 1
    t_res = ToolResult(success=True, message="Opened Chrome")
    v_res = VerificationResult(verified=True, message="Verified", details={})
    memory.record_success(step1, "open_application", ActionMethod.NATIVE_API, t_res, v_res, 120.0)

    # Next step should be step 2
    assert memory.current_step_index == 1
    assert memory.get_current_step() == step2
    assert memory.is_action_already_completed("open_application", "chrome") is True


def test_execution_memory_method_switch_threshold():
    step1 = ActionStep(step_id=1, description="Open app", intent_hint="open_application", target="app")
    plan = GoalPlan(goal_id="g1", original_goal="Open app", steps=[step1])
    memory = ExecutionMemory(original_goal="Open app", plan=plan)

    # Record 2 attempts of NATIVE_API
    memory.record_method_attempt(1, ActionMethod.NATIVE_API)
    memory.record_method_attempt(1, ActionMethod.NATIVE_API)

    failed_methods = memory.get_failed_methods_for_step(1, threshold=2)
    assert ActionMethod.NATIVE_API in failed_methods
