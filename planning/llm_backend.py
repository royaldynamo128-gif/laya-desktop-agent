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
    Analyzes natural speech connectors ('and', 'then', 'after that', ';', ',')
    and common computer operation patterns.
    """

    # Common multi-action connector patterns
    SPLIT_REGEX = re.compile(
        r"\b(?:and then|then|after that|and also|and)\b|;|, and|,",
        re.IGNORECASE,
    )

    def plan_goal(self, goal: str, context: Dict[str, Any]) -> GoalPlan:
        raw = goal.strip()
        goal_id = f"plan-{abs(hash(raw)) % 100000:05d}"

        # 1. Single action check
        clauses = self._split_clauses(raw)
        if len(clauses) <= 1:
            step = self._clause_to_step(1, raw, is_terminal=True)
            return GoalPlan(
                goal_id=goal_id,
                original_goal=raw,
                steps=[step],
                is_single_action=True,
                context_notes="Single-action direct execution",
            )

        # 2. Multi-step decomposition
        steps: List[ActionStep] = []
        for i, clause in enumerate(clauses, start=1):
            is_last = (i == len(clauses))
            step = self._clause_to_step(i, clause, is_terminal=is_last)
            steps.append(step)

        logger.info(
            "RuleBasedPlanner decomposed goal '%s' into %d steps: %s",
            raw,
            len(steps),
            [s.description for s in steps],
        )

        return GoalPlan(
            goal_id=goal_id,
            original_goal=raw,
            steps=steps,
            is_single_action=False,
            context_notes=f"Decomposed into {len(steps)} sub-actions",
        )

    def is_goal_complete(
        self, goal: str, completed_steps: List[str], current_state: Dict[str, Any]
    ) -> bool:
        """Heuristic goal completeness check."""
        clauses = self._split_clauses(goal)
        # If all clauses have a corresponding completed step, consider complete
        return len(completed_steps) >= len(clauses)

    def _split_clauses(self, text: str) -> List[str]:
        # Handle "Open X and write/type Y"
        # Protect specific quoted strings or URLs
        parts = [p.strip() for p in self.SPLIT_REGEX.split(text) if p.strip()]
        return parts if parts else [text]

    def _clause_to_step(self, step_id: int, clause: str, is_terminal: bool = False) -> ActionStep:
        c = clause.strip()
        lower = c.lower()

        # Heuristic intent matching
        if lower.startswith(("open ", "launch ", "start ")):
            target = re.sub(r"^(open|launch|start)\s+", "", c, flags=re.IGNORECASE).strip()
            return ActionStep(
                step_id=step_id,
                description=f"Open {target}",
                intent_hint="open_application",
                target=target,
                arguments={"app_name": target},
                verification_criteria=f"Application {target} is running and visible",
                is_terminal=is_terminal,
            )

        if lower.startswith(("close ", "quit ", "exit ")):
            target = re.sub(r"^(close|quit|exit)\s+", "", c, flags=re.IGNORECASE).strip()
            return ActionStep(
                step_id=step_id,
                description=f"Close {target}",
                intent_hint="close_application",
                target=target,
                arguments={"query": target},
                verification_criteria=f"Application {target} window is closed",
                is_terminal=is_terminal,
            )

        if lower.startswith(("switch to ", "focus ", "bring up ", "switch ")):
            target = re.sub(r"^(switch to|focus|bring up|switch)\s+", "", c, flags=re.IGNORECASE).strip()
            return ActionStep(
                step_id=step_id,
                description=f"Switch to {target}",
                intent_hint="switch_window",
                target=target,
                arguments={"query": target},
                verification_criteria=f"Window {target} is focused",
                is_terminal=is_terminal,
            )

        if lower.startswith(("type ", "write ", "enter text ", "enter ")):
            text_to_type = re.sub(r"^(type|write|enter text|enter)\s+", "", c, flags=re.IGNORECASE).strip()
            # Strip outer quotes if present
            if (text_to_type.startswith('"') and text_to_type.endswith('"')) or (
                text_to_type.startswith("'") and text_to_type.endswith("'")
            ):
                text_to_type = text_to_type[1:-1]
            return ActionStep(
                step_id=step_id,
                description=f"Type '{text_to_type}'",
                intent_hint="type_text",
                target=text_to_type,
                arguments={"text": text_to_type},
                verification_criteria=f"Text '{text_to_type}' entered",
                is_terminal=is_terminal,
            )

        if lower.startswith(("press ", "hit ", "key ")):
            key = re.sub(r"^(press|hit|key)\s+", "", c, flags=re.IGNORECASE).strip()
            return ActionStep(
                step_id=step_id,
                description=f"Press key {key}",
                intent_hint="press_key",
                target=key,
                arguments={"key": key},
                verification_criteria=f"Key {key} pressed",
                is_terminal=is_terminal,
            )

        if lower.startswith(("go to ", "visit ", "navigate to ")):
            url = re.sub(r"^(go to|visit|navigate to)\s+", "", c, flags=re.IGNORECASE).strip()
            return ActionStep(
                step_id=step_id,
                description=f"Navigate to {url}",
                intent_hint="browser_navigate",
                target=url,
                arguments={"url": url},
                verification_criteria=f"Browser navigated to {url}",
                is_terminal=is_terminal,
            )

        if lower.startswith(("search for ", "search ")):
            query = re.sub(r"^(search for|search)\s+", "", c, flags=re.IGNORECASE).strip()
            return ActionStep(
                step_id=step_id,
                description=f"Search for '{query}'",
                intent_hint="browser_search",
                target=query,
                arguments={"query": query, "engine": "google"},
                verification_criteria=f"Search performed for '{query}'",
                is_terminal=is_terminal,
            )

        if "screenshot" in lower:
            return ActionStep(
                step_id=step_id,
                description="Take screenshot",
                intent_hint="take_screenshot",
                target="screen",
                arguments={},
                verification_criteria="Screenshot captured",
                is_terminal=is_terminal,
            )

        if lower.startswith("scroll"):
            direction = "down" if "down" in lower else ("up" if "up" in lower else "down")
            return ActionStep(
                step_id=step_id,
                description=f"Scroll {direction}",
                intent_hint="scroll",
                target=direction,
                arguments={"direction": direction},
                verification_criteria=f"Scrolled {direction}",
                is_terminal=is_terminal,
            )

        # Fallback generic step
        return ActionStep(
            step_id=step_id,
            description=c,
            intent_hint="",
            target=c,
            arguments={"command": c},
            verification_criteria=f"Action '{c}' performed",
            is_terminal=is_terminal,
        )


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
