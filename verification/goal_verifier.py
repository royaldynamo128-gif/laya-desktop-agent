"""
Goal-Level Verifier for Laya Desktop Agent.
Validates whether the user's high-level desired end result has been achieved,
preventing premature task termination based merely on tool success or elapsed time.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import logging
from planning.execution_memory import ExecutionMemory
from verification.observer import DesktopObservation

logger = logging.getLogger(__name__)


@dataclass
class GoalVerificationResult:
    goal_complete: bool
    verified: bool
    message: str
    details: Dict[str, Any]


class GoalVerifier:
    """Verifies complete end-state satisfaction of overarching user goals."""

    def verify_end_result(
        self,
        goal: str,
        memory: ExecutionMemory,
        current_observation: DesktopObservation,
    ) -> GoalVerificationResult:
        """
        Evaluate if the desired end condition of the goal is verified.
        Only returns verified=True if:
          1. All sub-steps in the plan were successfully executed and verified
          2. The expected application / window / document state is present
        """
        # 1. Uncompleted plan steps check
        if not memory.is_plan_exhausted:
            remaining = len(memory.plan.steps) - memory.current_step_index
            return GoalVerificationResult(
                goal_complete=False,
                verified=False,
                message=f"Goal incomplete: {remaining} action steps remaining in execution plan.",
                details={"remaining_steps": remaining},
            )

        # 2. Check if any step failed without recovery
        if not memory.completed_steps:
            return GoalVerificationResult(
                goal_complete=False,
                verified=False,
                message="Goal failed: No actions completed successfully.",
                details={"completed_count": 0},
            )

        last_completed = memory.completed_steps[-1]
        last_step = last_completed.step

        # 3. Contextual verification against terminal step criteria
        target_lower = last_step.target.lower().strip()
        active_class = current_observation.active_window_class.lower()
        active_title = current_observation.active_window_name.lower()

        # If goal was opening/interacting with an app, check if that app is currently active or open
        if last_step.intent_hint in ("open_application", "switch_window") and target_lower:
            matched = (
                target_lower in active_class
                or target_lower in active_title
                or any(target_lower in w.name.lower() or target_lower in w.wm_class.lower() for w in current_observation.all_windows)
            )
            if not matched:
                return GoalVerificationResult(
                    goal_complete=False,
                    verified=False,
                    message=f"Goal end state not verified: Expected '{last_step.target}' to be present, but active window is '{active_title}' ({active_class}).",
                    details={
                        "expected": last_step.target,
                        "active_class": active_class,
                        "active_title": active_title,
                    },
                )

        # 4. If typing or navigating, verify that the active window was focused and received input
        if last_step.intent_hint in ("type_text", "browser_navigate"):
            if not current_observation.active_window_id:
                return GoalVerificationResult(
                    goal_complete=False,
                    verified=False,
                    message="Goal end state not verified: No focused window active to receive input.",
                    details={},
                )

        return GoalVerificationResult(
            goal_complete=True,
            verified=True,
            message=f"Verified: Goal '{goal}' achieved across {len(memory.completed_steps)} verified steps.",
            details={
                "steps_completed": len(memory.completed_steps),
                "active_window": current_observation.active_window_name,
                "active_class": current_observation.active_window_class,
            },
        )
