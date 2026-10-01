"""
Planning and Autonomous Goal Execution Subsystem for Laya Desktop Agent.
Decomposes natural language goals into verifiable multi-step action plans,
tracks execution memory, and orchestrates method fallbacks across the desktop.
"""

from planning.method_registry import ActionMethod, MethodRegistry
from planning.execution_memory import (
    ActionStep,
    GoalPlan,
    CompletedStep,
    FailedStep,
    ExecutionMemory,
)
from planning.llm_backend import LLMBackend, RuleBasedPlannerBackend
from planning.goal_planner import GoalPlanner
from planning.goal_executor import GoalExecutor, GoalResult

__all__ = [
    "ActionMethod",
    "MethodRegistry",
    "ActionStep",
    "GoalPlan",
    "CompletedStep",
    "FailedStep",
    "ExecutionMemory",
    "LLMBackend",
    "RuleBasedPlannerBackend",
    "GoalPlanner",
    "GoalExecutor",
    "GoalResult",
]
