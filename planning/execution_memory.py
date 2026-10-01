"""
Execution Memory and Short-Term State Tracking for Autonomous Multi-Step Tasks.
Maintains history of completed steps, failed steps, tried methods, current application/window,
and prevents repeating successfully completed actions.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Set
from planning.method_registry import ActionMethod
from tools.base import ToolResult
from verification.verifier import VerificationResult  # noqa: F401  (re-export; runtime-safe via deferred import)


@dataclass
class ActionStep:
    """An individual sub-goal or atomic action within a multi-step plan."""
    step_id: int
    description: str                     # Natural language step (e.g. "Open Google Chrome")
    intent_hint: str = ""                # Tool/intent hint (e.g. "open_application")
    target: str = ""                     # Target entity (e.g. "google chrome")
    arguments: Dict[str, Any] = field(default_factory=dict)
    verification_criteria: str = ""      # Expected condition (e.g. "Chrome window is visible")
    is_terminal: bool = False            # True if this step completes the overall goal


@dataclass
class CompletedStep:
    """Record of a successfully executed and verified step."""
    step: ActionStep
    tool_used: str
    method_used: ActionMethod
    result: ToolResult
    verification: VerificationResult
    elapsed_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class FailedStep:
    """Record of a failed execution or verification attempt."""
    step: ActionStep
    tool_used: str
    method_used: ActionMethod
    error_message: str
    attempt_number: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class GoalPlan:
    """Ordered sequence of steps designed to accomplish a user's goal."""
    goal_id: str
    original_goal: str
    steps: List[ActionStep] = field(default_factory=list)
    is_single_action: bool = False
    context_notes: str = ""


class ExecutionMemory:
    """Thread-safe short-term memory for an autonomous task lifecycle."""

    def __init__(self, original_goal: str, plan: Optional[GoalPlan] = None):
        self.original_goal = original_goal
        self.plan: GoalPlan = plan or GoalPlan(goal_id=str(int(time.time())), original_goal=original_goal)
        self.start_time = time.time()
        self.current_step_index = 0

        # State tracking
        self.current_application: str = ""
        self.current_window_id: Optional[str] = None
        self.current_window_name: str = ""
        self.current_window_class: str = ""
        self.current_ui_summary: str = ""

        # Step execution records
        self.completed_steps: List[CompletedStep] = []
        self.failed_attempts: List[FailedStep] = []

        # Method tracking per step_id: Dict[step_id, Dict[ActionMethod, int]] -> attempt count per method
        self._method_attempts_by_step: Dict[int, Dict[ActionMethod, int]] = {}

        # Set of action signatures already accomplished to prevent repetition
        self._completed_action_signatures: Set[str] = set()

    @property
    def total_elapsed_sec(self) -> float:
        return time.time() - self.start_time

    @property
    def is_plan_exhausted(self) -> bool:
        return self.current_step_index >= len(self.plan.steps)

    def get_current_step(self) -> Optional[ActionStep]:
        if 0 <= self.current_step_index < len(self.plan.steps):
            return self.plan.steps[self.current_step_index]
        return None

    def get_next_step(self) -> Optional[ActionStep]:
        next_idx = self.current_step_index + 1
        if next_idx < len(self.plan.steps):
            return self.plan.steps[next_idx]
        return None

    def record_method_attempt(self, step_id: int, method: ActionMethod) -> int:
        """Increment and return the consecutive attempt count for this method on this step."""
        if step_id not in self._method_attempts_by_step:
            self._method_attempts_by_step[step_id] = {}
        count = self._method_attempts_by_step[step_id].get(method, 0) + 1
        self._method_attempts_by_step[step_id][method] = count
        return count

    def get_method_attempt_count(self, step_id: int, method: ActionMethod) -> int:
        return self._method_attempts_by_step.get(step_id, {}).get(method, 0)

    def get_failed_methods_for_step(self, step_id: int, threshold: int = 2) -> List[ActionMethod]:
        """Return list of methods that have failed at or above the threshold count."""
        step_dict = self._method_attempts_by_step.get(step_id, {})
        return [m for m, count in step_dict.items() if count >= threshold]

    def record_success(
        self,
        step: ActionStep,
        tool_used: str,
        method_used: ActionMethod,
        result: ToolResult,
        verification: VerificationResult,
        elapsed_ms: float,
    ) -> None:
        """Record successful step completion and update signature registry."""
        entry = CompletedStep(
            step=step,
            tool_used=tool_used,
            method_used=method_used,
            result=result,
            verification=verification,
            elapsed_ms=elapsed_ms,
        )
        self.completed_steps.append(entry)
        sig = self._make_signature(step.intent_hint or tool_used, step.target)
        self._completed_action_signatures.add(sig)
        self.current_step_index += 1

    def record_failure(
        self,
        step: ActionStep,
        tool_used: str,
        method_used: ActionMethod,
        error_message: str,
    ) -> None:
        """Record failed execution attempt."""
        entry = FailedStep(
            step=step,
            tool_used=tool_used,
            method_used=method_used,
            error_message=error_message,
            attempt_number=len([f for f in self.failed_attempts if f.step.step_id == step.step_id]) + 1,
        )
        self.failed_attempts.append(entry)

    def is_action_already_completed(self, intent: str, target: str) -> bool:
        """Prevent repeating actions that have already been completed successfully."""
        sig = self._make_signature(intent, target)
        return sig in self._completed_action_signatures

    def update_desktop_state(
        self,
        active_window_id: Optional[str],
        active_window_name: str,
        active_window_class: str,
        ui_summary: str = "",
    ) -> None:
        """Update current active desktop focus state."""
        self.current_window_id = active_window_id
        self.current_window_name = active_window_name
        self.current_window_class = active_window_class
        if active_window_class:
            self.current_application = active_window_class
        elif active_window_name:
            self.current_application = active_window_name
        if ui_summary:
            self.current_ui_summary = ui_summary

    def _make_signature(self, intent: str, target: str) -> str:
        return f"{intent.lower().strip()}:{target.lower().strip()}"

    def to_summary_dict(self) -> Dict[str, Any]:
        """Diagnostic summary of execution memory."""
        return {
            "original_goal": self.original_goal,
            "total_steps": len(self.plan.steps),
            "completed_steps_count": len(self.completed_steps),
            "failed_attempts_count": len(self.failed_attempts),
            "current_step_index": self.current_step_index,
            "current_application": self.current_application,
            "current_window": self.current_window_name,
            "elapsed_sec": round(self.total_elapsed_sec, 2),
            "completed": [
                {
                    "step_id": c.step.step_id,
                    "desc": c.step.description,
                    "tool": c.tool_used,
                    "method": c.method_used.name,
                    "elapsed_ms": round(c.elapsed_ms, 1),
                }
                for c in self.completed_steps
            ],
            "failed": [
                {
                    "step_id": f.step.step_id,
                    "desc": f.step.description,
                    "tool": f.tool_used,
                    "method": f.method_used.name,
                    "error": f.error_message,
                }
                for f in self.failed_attempts
            ],
        }
