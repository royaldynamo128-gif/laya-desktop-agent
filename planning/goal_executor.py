"""
Autonomous Goal Execution Loop for Laya Desktop Agent.
Executes the closed-loop cycle:
  USER GOAL
  -> UNDERSTAND INTENT & PLAN
  -> OBSERVE CURRENT DESKTOP
  -> DETERMINE NEXT BEST ACTION
  -> EXECUTE ACTION (METHOD-AWARE)
  -> OBSERVE AGAIN
  -> VERIFY RESULT
  -> ERROR RECOVERY / METHOD SWITCHING (AFTER 2 FAILURES)
  -> CONTINUE UNTIL GOAL IS COMPLETE
"""

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.lifecycle import CommandLifecycle, CommandStage
from core.state import AgentState, StateManager
from execution.parameter_binder import BoundAction
from planning.execution_memory import (
    ActionStep,
    CompletedStep,
    ExecutionMemory,
    GoalPlan,
)
from planning.goal_planner import GoalPlanner
from planning.method_registry import ActionMethod, MethodRegistry
from tools.base import ToolResult
from verification.observer import DesktopObservation, DesktopObserver
from verification.verifier import ActionVerifier, VerificationResult

logger = logging.getLogger(__name__)


@dataclass
class GoalResult:
    """Final result of an autonomous goal execution."""
    goal: str
    success: bool
    verified: bool
    message: str
    total_steps_executed: int
    elapsed_ms: float
    memory_summary: Dict[str, Any]
    error: Optional[str] = None


class GoalExecutor:
    """
    Fully autonomous computer-use execution engine.
    Translates user intent into chained actions, performs continuous observation
    and verification, recovers from errors by switching methods, and terminates
    only when the user's desired end result is achieved.
    """

    def __init__(
        self,
        planner: GoalPlanner,
        coordinator: Any,  # Coordinator instance
        observer: DesktopObserver,
        verifier: ActionVerifier,
        method_registry: Optional[MethodRegistry] = None,
        max_replans: int = 3,
        method_switch_threshold: int = 2,
    ):
        self.planner = planner
        self.coordinator = coordinator
        self.observer = observer
        self.verifier = verifier
        self.method_registry = method_registry or MethodRegistry()
        self.max_replans = max_replans
        self.method_switch_threshold = method_switch_threshold

    def execute_goal(
        self,
        goal: str,
        lifecycle: Optional[CommandLifecycle] = None,
        speak_progress: bool = False,
    ) -> GoalResult:
        """
        Execute the complete autonomous loop until the user's requested goal is satisfied.
        Never stops after an intermediate action if the goal requires subsequent actions.
        """
        start_time = time.perf_counter()
        logger.info("=== STARTING AUTONOMOUS GOAL EXECUTION: '%s' ===", goal)

        # 1. OBSERVE CURRENT DESKTOP
        pre_obs = self.observer.observe(delay_sec=0.0)

        # 2. UNDERSTAND INTENT & BUILD PLAN
        if lifecycle:
            lifecycle.transition(CommandStage.PLAN_READY, detail=f"Planning: {goal}")

        plan = self.planner.plan(goal, desktop_state=pre_obs)
        memory = ExecutionMemory(original_goal=goal, plan=plan)
        memory.update_desktop_state(
            pre_obs.active_window_id,
            pre_obs.active_window_name,
            pre_obs.active_window_class,
        )

        replan_count = 0

        # FAST PATH: If goal is detected as a single action, delegate to single command pipeline
        if plan.is_single_action and len(plan.steps) == 1:
            logger.info("Single-action fast path detected for goal: '%s'", goal)
            single_res = self.coordinator.process_text_command(
                command_text=goal,
                lifecycle=lifecycle,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return GoalResult(
                goal=goal,
                success=single_res.success,
                verified=single_res.verified,
                message=single_res.response_text,
                total_steps_executed=1,
                elapsed_ms=elapsed_ms,
                memory_summary=memory.to_summary_dict(),
                error=None if single_res.success else single_res.response_text,
            )

        # 3. AUTONOMOUS MULTI-STEP EXECUTION LOOP
        while not memory.is_plan_exhausted:
            # Check Emergency Stop latch
            if hasattr(self.coordinator, "emergency_stop") and self.coordinator.emergency_stop.is_stopped():
                logger.critical("Emergency stop active. Aborting goal execution.")
                if lifecycle:
                    lifecycle.cancel(reason="Emergency Stop Activated")
                return self._finalize_result(
                    goal, False, False, "Emergency Stop Activated", memory, start_time, error="Emergency Stop"
                )

            step = memory.get_current_step()
            if not step:
                break

            total_steps = len(memory.plan.steps)
            step_num = step.step_id
            logger.info(">>> STEP %d/%d: %s <<<", step_num, total_steps, step.description)

            # Update UI status overlay if present
            if hasattr(self.coordinator, "status_broadcaster") and self.coordinator.status_broadcaster:
                self.coordinator.status_broadcaster.broadcast(
                    state="EXECUTING",
                    title=f"Step {step_num}/{total_steps}: {step.description}",
                    detail=f"Goal: {goal}",
                    next_action=step.verification_criteria,
                    command_text=goal,
                )

            # Execute step with method fallback
            step_success, step_verified, error_msg = self._execute_step_with_fallbacks(
                step=step,
                memory=memory,
                lifecycle=lifecycle,
            )

            # Re-observe desktop state after step
            post_obs = self.observer.observe(delay_sec=0.1)
            memory.update_desktop_state(
                post_obs.active_window_id,
                post_obs.active_window_name,
                post_obs.active_window_class,
            )

            if not step_verified:
                logger.warning("Step %d ('%s') failed verification.", step_num, step.description)
                if replan_count < self.max_replans:
                    replan_count += 1
                    logger.info("Initiating dynamic replan %d/%d...", replan_count, self.max_replans)
                    new_plan = self.planner.replan(goal, memory, failed_step=step, desktop_state=post_obs)
                    memory.plan = new_plan
                    memory.current_step_index = 0
                    continue
                else:
                    logger.error("Max replans (%d) exhausted for goal '%s'.", self.max_replans, goal)
                    msg = f"Failed to complete goal at step '{step.description}': {error_msg}"
                    return self._finalize_result(goal, False, False, msg, memory, start_time, error=error_msg)

        # 4. FINAL GOAL-LEVEL VERIFICATION
        final_obs = self.observer.observe(delay_sec=0.15)
        goal_complete, goal_msg = self.planner.is_goal_complete(goal, memory, final_obs)

        if goal_complete:
            logger.info("Goal '%s' successfully completed and verified: %s", goal, goal_msg)
            if lifecycle:
                lifecycle.succeed(detail=goal_msg)
            return self._finalize_result(goal, True, True, goal_msg, memory, start_time)
        else:
            logger.warning("Goal '%s' finished all steps but goal-level verification failed: %s", goal, goal_msg)
            if lifecycle:
                lifecycle.fail(reason=goal_msg)
            return self._finalize_result(goal, True, False, goal_msg, memory, start_time, error=goal_msg)

    def _execute_step_with_fallbacks(
        self,
        step: ActionStep,
        memory: ExecutionMemory,
        lifecycle: Optional[CommandLifecycle] = None,
    ) -> Tuple[bool, bool, str]:
        """
        Execute an ActionStep using method switching.
        After 2 unsuccessful attempts with a method, automatically switch to the next method.
        """
        action_name = step.intent_hint or "generic"
        methods = self.method_registry.get_methods_for_action(action_name)
        failed_methods: List[ActionMethod] = []

        last_error = "Unknown error"

        while True:
            current_method = self.method_registry.get_next_method(action_name, failed_methods)
            if current_method is None:
                logger.error("All methods exhausted for step '%s'.", step.description)
                return False, False, f"All interaction methods exhausted: {last_error}"

            # Attempt execution with current method (up to method_switch_threshold attempts)
            for attempt in range(1, self.method_switch_threshold + 1):
                memory.record_method_attempt(step.step_id, current_method)
                logger.info(
                    "Executing step %d via [%s] (attempt %d/%d)...",
                    step.step_id,
                    current_method.name,
                    attempt,
                    self.method_switch_threshold,
                )

                step_start = time.perf_counter()
                pre_obs = self.observer.observe(delay_sec=0.0)

                # Bind & Execute
                tool_res, bound_action = self._dispatch_action(step, current_method)
                step_elapsed_ms = (time.perf_counter() - step_start) * 1000.0

                if not tool_res.success:
                    last_error = tool_res.error or tool_res.message
                    memory.record_failure(step, bound_action.tool_name if bound_action else action_name, current_method, last_error)
                    continue

                # Verify step
                time.sleep(0.1)
                post_obs = self.observer.observe(delay_sec=0.05)
                verify_res = self._verify_step(step, bound_action, tool_res, pre_obs, post_obs)

                if verify_res.verified:
                    logger.info("Step %d verified successfully via [%s]!", step.step_id, current_method.name)
                    memory.record_success(
                        step=step,
                        tool_used=bound_action.tool_name if bound_action else action_name,
                        method_used=current_method,
                        result=tool_res,
                        verification=verify_res,
                        elapsed_ms=step_elapsed_ms,
                    )
                    return True, True, ""

                last_error = verify_res.message
                memory.record_failure(
                    step,
                    bound_action.tool_name if bound_action else action_name,
                    current_method,
                    last_error,
                )

            # Method failed after threshold attempts -> mark as failed and switch
            logger.warning(
                "Method [%s] failed %d times for step %d. Switching methods...",
                current_method.name,
                self.method_switch_threshold,
                step.step_id,
            )
            failed_methods.append(current_method)

    def _dispatch_action(
        self, step: ActionStep, method: ActionMethod
    ) -> Tuple[ToolResult, Optional[BoundAction]]:
        """Dispatch step to the appropriate tool execution mechanism."""
        intent = step.intent_hint.lower().strip()
        target = step.target.strip()
        args = dict(step.arguments)

        # 1. Open Application
        if intent in ("open_application", "launch", "open"):
            bound = None
            if hasattr(self.coordinator, "binder") and self.coordinator.binder:
                bound = self.coordinator.binder.bind(f"open {target}", None)
            if not bound:
                bound = BoundAction(
                    tool_name="open_application",
                    arguments={"name": target, "app_name": target},
                    target=target,
                )
            res = self.coordinator.executor.execute_action(bound)
            return res, bound

        # 2. Close Application
        if intent in ("close_application", "close", "quit"):
            bound = BoundAction(tool_name="close_application", arguments={"query": target}, target=target)
            res = self.coordinator.executor.execute_action(bound)
            return res, bound

        # 3. Switch Window
        if intent in ("switch_window", "switch", "focus"):
            bound = BoundAction(tool_name="switch_window", arguments={"query": target}, target=target)
            res = self.coordinator.executor.execute_action(bound)
            return res, bound

        # 4. Type Text
        if intent in ("type_text", "type", "write"):
            text = args.get("text", target)
            bound = BoundAction(tool_name="type_text", arguments={"text": text}, target=text)
            res = self.coordinator.executor.execute_action(bound)
            return res, bound

        # 5. Press Key
        if intent in ("press_key", "key", "press"):
            key = args.get("key", target)
            bound = BoundAction(tool_name="press_key", arguments={"key": key}, target=key)
            res = self.coordinator.executor.execute_action(bound)
            return res, bound

        # 6. Browser Search & Navigate
        if intent in ("browser_search", "search"):
            query = args.get("query") or target
            engine = args.get("engine", "google")
            bound = BoundAction(tool_name="browser_search", arguments={"query": query, "engine": engine}, target=query)
            res = self.coordinator.executor.execute_action(bound)
            return res, bound

        if intent in ("browser_navigate", "navigate"):
            url = args.get("url") or target
            bound = BoundAction(tool_name="browser_navigate", arguments={"url": url}, target=url)
            res = self.coordinator.executor.execute_action(bound)
            return res, bound

        # 7. Scroll
        if intent in ("scroll", "scroll_down", "scroll_up"):
            direction = args.get("direction", "down")
            bound = BoundAction(tool_name="scroll", arguments={"direction": direction}, target=direction)
            res = self.coordinator.executor.execute_action(bound)
            return res, bound

        # 8. Screenshot
        if intent in ("take_screenshot", "screenshot"):
            bound = BoundAction(tool_name="take_screenshot", arguments={}, target="screen")
            res = self.coordinator.executor.execute_action(bound)
            return res, bound

        # Generic: fallback to coordinator text command parser
        logger.info("Dispatching step '%s' to coordinator text command...", step.description)
        pipe_res = self.coordinator.process_text_command(step.description)
        fake_bound = BoundAction(tool_name=pipe_res.action_executed or "generic", arguments={}, target=step.target)
        tool_res = ToolResult(success=pipe_res.success, message=pipe_res.response_text)
        return tool_res, fake_bound

    def _verify_step(
        self,
        step: ActionStep,
        bound_action: Optional[BoundAction],
        tool_res: ToolResult,
        pre_obs: DesktopObservation,
        post_obs: DesktopObservation,
    ) -> VerificationResult:
        """Verify whether an individual step succeeded."""
        if not tool_res.success:
            return VerificationResult(
                verified=False,
                message=tool_res.error or tool_res.message,
                details={"stage": "execution_failed"},
            )

        if bound_action:
            v_res = self.verifier.verify(bound_action, tool_res, pre_obs, post_obs)
            return v_res

        return VerificationResult(verified=True, message="Step completed.", details={})

    def _finalize_result(
        self,
        goal: str,
        success: bool,
        verified: bool,
        message: str,
        memory: ExecutionMemory,
        start_time: float,
        error: Optional[str] = None,
    ) -> GoalResult:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return GoalResult(
            goal=goal,
            success=success,
            verified=verified,
            message=message,
            total_steps_executed=len(memory.completed_steps),
            elapsed_ms=elapsed_ms,
            memory_summary=memory.to_summary_dict(),
            error=error,
        )
