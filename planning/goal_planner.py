"""
Goal Planner for Laya Desktop Agent.
Orchestrates goal understanding, multi-step decomposition, and dynamic replanning
using pluggable LLM and heuristic backends.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
from planning.execution_memory import ActionStep, ExecutionMemory, GoalPlan
from planning.llm_backend import AutoPlannerBackend, LLMBackend, RuleBasedPlannerBackend
from verification.observer import DesktopObservation

logger = logging.getLogger(__name__)


class GoalPlanner:
    """High-level planner decomposing user intentions into verifiable desktop steps."""

    def __init__(self, backend: Optional[LLMBackend] = None):
        self.backend = backend or AutoPlannerBackend(primary_backend=None)

    def plan(self, goal: str, desktop_state: Optional[DesktopObservation] = None) -> GoalPlan:
        """Decompose a user goal into an ordered sequence of ActionSteps."""
        ctx = self._build_context(desktop_state)
        logger.info("Decomposing goal: '%s' (Active window: %s)", goal, ctx.get("active_window", "None"))
        plan = self.backend.plan_goal(goal, ctx)
        logger.info("Goal plan generated: %d steps (is_single_action=%s)", len(plan.steps), plan.is_single_action)
        return plan

    def replan(
        self,
        goal: str,
        memory: ExecutionMemory,
        failed_step: ActionStep,
        desktop_state: Optional[DesktopObservation] = None,
    ) -> GoalPlan:
        """
        Dynamically adjust or regenerate remaining plan when a step cannot be recovered.
        Preserves all completed steps and generates alternatives for the remainder.
        """
        logger.warning(
            "Replanning goal '%s' after failure of step %d ('%s')",
            goal,
            failed_step.step_id,
            failed_step.description,
        )
        ctx = self._build_context(desktop_state)
        ctx["failed_step"] = failed_step.description
        ctx["completed_steps"] = [c.step.description for c in memory.completed_steps]

        # Generate fresh plan from current desktop state
        raw_new_plan = self.backend.plan_goal(goal, ctx)

        # Filter out steps that are already completed
        filtered_steps: List[ActionStep] = []
        step_counter = len(memory.completed_steps) + 1
        for s in raw_new_plan.steps:
            if not memory.is_action_already_completed(s.intent_hint, s.target):
                s.step_id = step_counter
                filtered_steps.append(s)
                step_counter += 1

        if not filtered_steps:
            # If no steps remained, create a fallback direct interaction step
            filtered_steps.append(
                ActionStep(
                    step_id=step_counter,
                    description=f"Retry action for: {goal}",
                    intent_hint=failed_step.intent_hint,
                    target=failed_step.target,
                    is_terminal=True,
                )
            )

        new_plan = GoalPlan(
            goal_id=f"replan-{memory.plan.goal_id}",
            original_goal=goal,
            steps=filtered_steps,
            is_single_action=len(filtered_steps) == 1,
            context_notes=f"Replanned after step {failed_step.step_id} failed",
        )
        return new_plan

    def is_goal_complete(
        self,
        goal: str,
        memory: ExecutionMemory,
        desktop_state: Optional[DesktopObservation] = None,
    ) -> Tuple[bool, str]:
        """
        Verify if the end result condition of the original goal is satisfied.
        Stop condition: Only declare task complete when requested final state is verified.
        """
        # If there are pending uncompleted steps in the active plan, not complete
        if not memory.is_plan_exhausted:
            current_step = memory.get_current_step()
            desc = current_step.description if current_step else "unknown"
            return False, f"In progress: Pending step '{desc}'"

        # Check if backend confirms completion
        completed_descs = [c.step.description for c in memory.completed_steps]
        ctx = self._build_context(desktop_state)
        complete = self.backend.is_goal_complete(goal, completed_descs, ctx)

        if complete:
            return True, f"Goal '{goal}' verified complete across {len(memory.completed_steps)} steps."
        return False, f"Goal '{goal}' end state could not be verified."

    def _build_context(self, desktop_state: Optional[DesktopObservation]) -> Dict[str, Any]:
        if not desktop_state:
            return {}
        return {
            "active_window": desktop_state.active_window_name,
            "active_class": desktop_state.active_window_class,
            "desktop_number": desktop_state.desktop_number,
            "open_window_count": len(desktop_state.open_window_ids),
        }
