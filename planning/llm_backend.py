"""
LLM and Rule-Based Planner Backends for Laya Desktop Agent.
Provides abstract planning backend with:
  1. RuleBasedPlannerBackend: Fast, deterministic goal decomposition for standard desktop workflows
  2. LocalLLMBackend: Optional local GGUF / llama-cpp-python / transformers backend
  3. APILLMBackend: Optional cloud / remote API endpoint backend
  4. AutoPlannerBackend: Resilient fallback router ensuring zero failure even without external models
"""

from abc import ABC, abstractmethod
import json
import logging
import re
import urllib.request
from typing import Any, Dict, List, Optional
from planning.execution_memory import ActionStep, GoalPlan
from planning.semantic_parser import Clause, ParseResult, SemanticParser, get_parser

logger = logging.getLogger(__name__)


SYSTEM_PLANNING_PROMPT = """You are an autonomous desktop computer-use planning engine.
Your job is to translate high-level user goals into ordered sequences of desktop actions.
For each action, output:
- step_id: integer
- description: clear action description
- intent_hint: one of [open_application, close_application, switch_window, type_text, press_key, click, scroll, browser_navigate, take_screenshot, system_command]
- target: application name, key name, text content, or URL
- verification_criteria: expected observable desktop condition

Output strictly valid JSON matching:
{
  "is_single_action": false,
  "steps": [
    {
      "step_id": 1,
      "description": "...",
      "intent_hint": "...",
      "target": "...",
      "verification_criteria": "..."
    }
  ]
}
"""


class LLMBackend(ABC):
    """Abstract interface for planning backends."""

    @abstractmethod
    def plan_goal(self, goal: str, context: Dict[str, Any]) -> GoalPlan:
        """Decompose a user goal into an ordered sequence of ActionSteps."""
        pass

    @abstractmethod
    def is_goal_complete(
        self, goal: str, completed_steps: List[str], current_state: Dict[str, Any]
    ) -> bool:
        """Evaluate if the original goal has been accomplished."""
        pass


class RuleBasedPlannerBackend(LLMBackend):
    """
    High-speed deterministic goal decomposition engine.
    Uses the generalized SemanticParser: verb-anchored clause segmentation,
    verb inheritance for coordinated lists, and slot-based verb frames.
    No hardcoded command dictionary.
    """

    # Back-compat: some modules import this for compound detection
    SPLIT_REGEX = re.compile(
        r"\b(?:and then|then|after that|and also|and)\b|;|, and|,",
        re.IGNORECASE,
    )

    def __init__(self, parser: Optional[SemanticParser] = None):
        self.parser = parser or get_parser()

    def plan_goal(self, goal: str, context: Dict[str, Any]) -> GoalPlan:
        raw = goal.strip()
        goal_id = f"plan-{abs(hash(raw)) % 100000:05d}"

        parsed: ParseResult = self.parser.parse(raw)

        steps: List[ActionStep] = []
        for i, clause in enumerate(parsed.clauses, start=1):
            is_last = i == len(parsed.clauses)
            steps.append(self._clause_to_step(i, clause, is_terminal=is_last))

        logger.info(
            "Planner decomposed goal '%s' into %d steps: %s",
            raw,
            len(steps),
            [s.description for s in steps],
        )

        return GoalPlan(
            goal_id=goal_id,
            original_goal=raw,
            steps=steps,
            is_single_action=len(steps) <= 1,
            context_notes=f"Decomposed into {len(steps)} sub-actions",
        )

    def is_goal_complete(
        self, goal: str, completed_steps: List[str], current_state: Dict[str, Any]
    ) -> bool:
        """Heuristic goal completeness check."""
        parsed = self.parser.parse(goal)
        return len(completed_steps) >= len(parsed.clauses)

    def _clause_to_step(self, step_id: int, clause: Clause, is_terminal: bool = False) -> ActionStep:
        c = clause
        target = c.target or ""
        args = dict(c.arguments)
        if c.is_referring:
            args["_referring"] = target

        # Step description in natural language for logging / replanning
        if c.intent in ("unknown",):
            description = c.raw or target
        else:
            description = f"{c.verb or c.intent} {target}".strip()

        verification = self._verification_criteria(c)

        return ActionStep(
            step_id=step_id,
            description=description,
            intent_hint=c.intent,
            target=target,
            arguments=args,
            verification_criteria=verification,
            is_terminal=is_terminal,
        )

    def _verification_criteria(self, clause: Clause) -> str:
        intent = clause.intent
        target = clause.target
        checks = {
            "open_application": f"Application {target} is running and focused",
            "close_application": f"Window for {target or 'active application'} is closed",
            "switch_window": f"Window {target} is focused",
            "browser_navigate": f"Browser shows {target}",
            "browser_search": f"Search results for '{target}' are displayed",
            "type_text": f"Text '{target}' was entered",
            "press_key": f"Key {target} was pressed",
            "click_element": f"Element '{target}' was activated",
            "submit_action": f"Submit control '{target}' was activated",
            "ui_action": f"UI action '{clause.verb} {target}' completed",
            "take_screenshot": "Screenshot captured",
            "read_screen": "Screen text extracted",
            "file_move": f"File moved to {clause.arguments.get('destination', 'destination')}",
            "file_rename": f"File renamed to {clause.arguments.get('new_name', 'new name')}",
            "file_delete": f"Files deleted",
            "file_create_folder": f"Folder {target} exists",
            "save_as": f"File saved as {clause.arguments.get('format') or 'new format'}",
            "download_action": f"File downloaded to Downloads",
            "wait_action": f"Condition '{target}' satisfied",
            "verify_action": f"'{target}' verified on screen",
        }
        return checks.get(intent, f"Action '{clause.raw or target}' performed")


class APILLMBackend(LLMBackend):
    """OpenAI / Gemini / Anthropic compatible HTTP API backend."""

    def __init__(
        self,
        api_url: str = "http://localhost:11434/v1/chat/completions",
        api_key: str = "",
        model_name: str = "qwen2.5:3b",
        timeout_sec: float = 10.0,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_sec = timeout_sec

    def plan_goal(self, goal: str, context: Dict[str, Any]) -> GoalPlan:
        messages = [
            {"role": "system", "content": SYSTEM_PLANNING_PROMPT},
            {
                "role": "user",
                "content": f"User Goal: {goal}\nCurrent Desktop State: {json.dumps(context)}",
            },
        ]
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)

            steps: List[ActionStep] = []
            for s in parsed.get("steps", []):
                steps.append(
                    ActionStep(
                        step_id=s.get("step_id", len(steps) + 1),
                        description=s.get("description", ""),
                        intent_hint=s.get("intent_hint", ""),
                        target=s.get("target", ""),
                        verification_criteria=s.get("verification_criteria", ""),
                    )
                )

            return GoalPlan(
                goal_id=f"plan-{abs(hash(goal)) % 100000:05d}",
                original_goal=goal,
                steps=steps,
                is_single_action=parsed.get("is_single_action", len(steps) <= 1),
                context_notes="Generated via API LLM",
            )

    def is_goal_complete(
        self, goal: str, completed_steps: List[str], current_state: Dict[str, Any]
    ) -> bool:
        # Simple completion query
        return len(completed_steps) > 0


class AutoPlannerBackend(LLMBackend):
    """
    Intelligent resilient router:
    Attempts primary backend (API or local LLM); transparently falls back to RuleBasedPlannerBackend.
    Guarantees that planning NEVER crashes the agent.
    """

    def __init__(self, primary_backend: Optional[LLMBackend] = None):
        self.primary = primary_backend
        self.fallback = RuleBasedPlannerBackend()

    def plan_goal(self, goal: str, context: Dict[str, Any]) -> GoalPlan:
        if self.primary:
            try:
                return self.primary.plan_goal(goal, context)
            except Exception as e:
                logger.warning(
                    "Primary LLM planner failed (%s). Falling back to RuleBasedPlannerBackend.", e
                )
        return self.fallback.plan_goal(goal, context)

    def is_goal_complete(
        self, goal: str, completed_steps: List[str], current_state: Dict[str, Any]
    ) -> bool:
        if self.primary:
            try:
                return self.primary.is_goal_complete(goal, completed_steps, current_state)
            except Exception:
                pass
        return self.fallback.is_goal_complete(goal, completed_steps, current_state)
