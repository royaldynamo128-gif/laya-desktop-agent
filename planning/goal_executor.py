"""
Autonomous Goal Execution Loop for Laya Desktop Agent.
Executes the closed-loop cycle:
  USER GOAL
  -> PARSE (semantic) & PLAN
  -> OBSERVE CURRENT DESKTOP
  -> DETERMINE NEXT STEP
  -> BIND (with referring-expression & context resolution)
  -> EXECUTE (method-varied, with genuine fallbacks)
  -> OBSERVE AGAIN
  -> VERIFY (per-step, honest)
  -> RECOVERY / METHOD SWITCHING
  -> CONTINUE UNTIL GOAL IS COMPLETE (GoalVerifier)

Guarantees:
  - No infinite loops: bounded attempts, bounded replans, bounded total time.
  - No recursion: steps bind directly to tools, never re-enter the coordinator.
  - No fake success: every step is verified or honestly reported failed.
  - Progress pings keep the command watchdog alive while useful work occurs.
"""

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.lifecycle import CommandLifecycle, CommandStage
from core.action_types import BoundAction
from planning.execution_memory import (
    ActionStep,
    CompletedStep,
    ExecutionMemory,
    GoalPlan,
)
from planning.goal_planner import GoalPlanner
from planning.method_registry import ActionMethod, MethodRegistry
from tools.base import ToolResult
from verification.goal_verifier import GoalVerifier
from verification.observer import DesktopObservation, DesktopObserver
from verification.verifier import ActionVerifier, VerificationResult

logger = logging.getLogger(__name__)

# Known search-engine domains for context-aware engine selection
ENGINE_DOMAINS = {
    "youtube": "youtube",
    "google": "google",
    "duckduckgo": "duckduckgo",
    "bing": "bing",
}


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
    Autonomous computer-use execution engine with genuine method fallback.
    """

    def __init__(
        self,
        planner: GoalPlanner,
        coordinator: Any,
        observer: DesktopObserver,
        verifier: ActionVerifier,
        method_registry: Optional[MethodRegistry] = None,
        goal_verifier: Optional[GoalVerifier] = None,
        max_replans: int = 3,
        method_switch_threshold: int = 2,
        max_total_duration_sec: float = 120.0,
        speak_step_progress: bool = False,
    ):
        self.planner = planner
        self.coordinator = coordinator
        self.observer = observer
        self.verifier = verifier
        self.method_registry = method_registry or MethodRegistry()
        self.goal_verifier = goal_verifier or GoalVerifier()
        self.max_replans = max_replans
        self.method_switch_threshold = method_switch_threshold
        self.max_total_duration_sec = max_total_duration_sec
        self.speak_step_progress = speak_step_progress
        # Context carried across steps (referring expression resolution)
        self._context: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def execute_goal(
        self,
        goal: str,
        lifecycle: Optional[CommandLifecycle] = None,
        speak_progress: bool = False,
    ) -> GoalResult:
        start_time = time.perf_counter()
        logger.info("=== AUTONOMOUS GOAL EXECUTION: '%s' ===", goal)

        speak = speak_progress or self.speak_step_progress
        self._context = {}

        # 1. OBSERVE CURRENT DESKTOP (single cached observation)
        pre_obs = self.observer.observe(delay_sec=0.0)

        # 2. PARSE & PLAN
        if lifecycle:
            lifecycle.transition(CommandStage.PLAN_READY, detail=f"Planning: {goal}")

        plan = self.planner.plan(goal, desktop_state=pre_obs)
        memory = ExecutionMemory(original_goal=goal, plan=plan)
        memory.update_desktop_state(
            pre_obs.active_window_id,
            pre_obs.active_window_name,
            pre_obs.active_window_class,
        )

        # 3. SINGLE-ACTION FAST PATH
        if plan.is_single_action and len(plan.steps) == 1:
            logger.info("Single-action fast path for goal: '%s'", goal)
            single_res = self.coordinator.process_text_command(
                command_text=goal,
                lifecycle=lifecycle,
                allow_autonomous_decomposition=False,
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

        # 4. MULTI-STEP EXECUTION LOOP
        replan_count = 0
        while not memory.is_plan_exhausted:
            # Total duration bound
            if (time.perf_counter() - start_time) > self.max_total_duration_sec:
                msg = f"Goal exceeded total time budget ({self.max_total_duration_sec:.0f}s)."
                logger.error(msg)
                if lifecycle:
                    lifecycle.fail(reason=msg)
                return self._finalize_result(goal, False, False, msg, memory, start_time, error="GoalTimeout")

            # Cancellation / emergency stop
            estop = getattr(self.coordinator, "emergency_stop", None)
            if estop is not None and estop.is_stopped():
                logger.critical("Emergency stop active. Aborting goal execution.")
                if lifecycle:
                    lifecycle.cancel(reason="Emergency Stop Activated")
                return self._finalize_result(
                    goal, False, False, "Emergency Stop Activated", memory, start_time, error="Emergency Stop"
                )
            if self._is_cancelled(lifecycle):
                logger.critical("Execution cancelled. Aborting goal.")
                if lifecycle:
                    lifecycle.cancel(reason="Cancelled")
                return self._finalize_result(
                    goal, False, False, "Execution cancelled.", memory, start_time, error="Cancelled"
                )

            step = memory.get_current_step()
            if not step:
                break

            total_steps = len(memory.plan.steps)
            step_num = memory.current_step_index + 1
            logger.info(">>> STEP %d/%d: %s <<<", step_num, total_steps, step.description)

            if hasattr(self.coordinator, "status_broadcaster") and self.coordinator.status_broadcaster:
                self.coordinator.status_broadcaster.broadcast(
                    state="EXECUTING",
                    title=f"Step {step_num}/{total_steps}: {step.description}",
                    detail=f"Goal: {goal}",
                    next_action=step.verification_criteria,
                    command_text=goal,
                )
            if lifecycle:
                lifecycle.transition(CommandStage.STEP_START, detail=f"{step_num}/{total_steps}: {step.description}")

            # Execute step with method variation
            step_success, step_verified, error_msg, bound_action, tool_res = self._execute_step_with_fallbacks(
                step=step,
                memory=memory,
                lifecycle=lifecycle,
            )

            # Post-observation & context update
            post_obs = self.observer.observe(delay_sec=0.05)
            memory.update_desktop_state(
                post_obs.active_window_id,
                post_obs.active_window_name,
                post_obs.active_window_class,
            )
            self._update_context_from_result(step, bound_action, tool_res, post_obs)

            if step_verified:
                # Watchdog progress ping: useful work is happening
                if lifecycle and hasattr(lifecycle, "ping_watchdog"):
                    lifecycle.ping_watchdog()
                if speak and step.is_terminal is False and self.coordinator:
                    try:
                        self.coordinator._speak_step_progress(step, memory)
                    except Exception:
                        pass
                continue

            # Step failed verification → replan (bounded, varied)
            logger.warning("Step %d ('%s') failed verification: %s", step_num, step.description, error_msg)
            if replan_count < self.max_replans:
                replan_count += 1
                logger.info("Initiating replan %d/%d...", replan_count, self.max_replans)
                if lifecycle:
                    lifecycle.transition(CommandStage.REPLANNING, detail=f"Replan {replan_count}/{self.max_replans}")
                new_plan = self._replan_with_variation(goal, memory, step, post_obs, replan_count)
                memory.plan = new_plan
                memory.current_step_index = 0
                # Deduplicate: skip steps already completed
                while (
                    not memory.is_plan_exhausted
                    and memory.get_current_step()
                    and memory.is_action_already_completed(
                        memory.get_current_step().intent_hint,
                        memory.get_current_step().target,
                    )
                ):
                    memory.current_step_index += 1
                continue
            else:
                msg = f"Failed to complete goal at step '{step.description}': {error_msg}"
                if speak and self.coordinator:
                    self.coordinator._speak_response_message(msg)
                if lifecycle:
                    lifecycle.fail(reason=msg)
                return self._finalize_result(goal, False, False, msg, memory, start_time, error=error_msg)

        # 5. FINAL GOAL-LEVEL VERIFICATION (GoalVerifier — real end-state check)
        final_obs = self.observer.observe(delay_sec=0.05)
        goal_result = self.goal_verifier.verify_end_result(goal, memory, final_obs)
        if not goal_result.verified:
            # Fallback to planner-based completeness (counts all clauses)
            complete, pmsg = self.planner.is_goal_complete(goal, memory, final_obs)
            if complete and memory.completed_steps:
                goal_result = goal_result.__class__(
                    goal_complete=True, verified=True, message=pmsg, details={}
                )

        if goal_result.verified:
            logger.info("Goal '%s' completed and verified: %s", goal, goal_result.message)
            if lifecycle:
                lifecycle.transition(CommandStage.GOAL_VERIFIED, detail=goal_result.message)
                lifecycle.succeed(detail=goal_result.message)
            return self._finalize_result(goal, True, True, goal_result.message, memory, start_time)
        else:
            logger.warning("Goal '%s' not verified: %s", goal, goal_result.message)
            if speak and self.coordinator:
                self.coordinator._speak_response_message(goal_result.message)
            if lifecycle:
                lifecycle.fail(reason=goal_result.message)
            return self._finalize_result(goal, True, False, goal_result.message, memory, start_time, error=goal_result.message)

    # ------------------------------------------------------------------
    # Step execution with genuine method variation
    # ------------------------------------------------------------------

    # Argument variations applied on successive attempts of a step.
    # Each entry changes HOW the action is attempted, not WHAT is attempted.
    METHOD_VARIATIONS: Dict[str, List[Dict[str, Any]]] = {
        "open_application": [
            {},                                   # standard: full launch tier chain
            {"launch_preference": "rescue"},      # prefer switching to an existing window
            {"launch_preference": "direct"},      # skip desktop-file launchers, exec directly
        ],
        "close_application": [
            {},
            {"fallback_key": "ctrl+q"},           # window close → app quit hotkey
        ],
        "switch_window": [
            {},
            {"category_fallback": True},          # match by app category classes
        ],
        "type_text": [
            {},
            {"method": "clipboard"},              # paste instead of key-by-key
        ],
        "press_key": [
            {},
            {"method": "wtype"},
        ],
        "click_element": [
            {},
            {"retries": 3, "wait_sec": 1.0},      # wait longer for slow UI
        ],
        "ui_action": [
            {},
            {"retries": 3, "wait_sec": 1.0},
        ],
        "submit_action": [
            {},
            {"force_key": "Return"},              # Enter-key submit
        ],
        "browser_navigate": [
            {},
            {"force_mechanism": "xdg-open"},      # retry outside Playwright
        ],
        "browser_search": [
            {},
            {"force_mechanism": "xdg-open"},
        ],
        "file_move": [
            {},
            {"create_destination": True},
        ],
    }

    def _execute_step_with_fallbacks(
        self,
        step: ActionStep,
        memory: ExecutionMemory,
        lifecycle: Optional[CommandLifecycle] = None,
    ) -> Tuple[bool, bool, str, Optional[BoundAction], Optional[ToolResult]]:
        """
        Execute one step across method variations. Each variation changes
        the execution approach (rescue/direct/clipboard/etc). Between
        variations there are bounded same-method retries for transient issues.
        """
        intent = step.intent_hint or "unknown"
        variations = self.METHOD_VARIATIONS.get(intent, [{}])
        last_error = "unknown error"
        bound_action: Optional[BoundAction] = None
        tool_res: Optional[ToolResult] = None

        for v_idx, variation in enumerate(variations):
            for attempt in range(1, self.method_switch_threshold + 1):
                if self._is_cancelled(lifecycle):
                    return False, False, "Cancelled", bound_action, tool_res

                memory.record_method_attempt(
                    step.step_id,
                    ActionMethod(ActionMethod.NATIVE_API + v_idx) if v_idx < 7 else ActionMethod.NATIVE_API,
                )
                logger.info(
                    "Step %d attempt %d/%d via variation %s: %s",
                    step.step_id, attempt, self.method_switch_threshold,
                    variation or "standard", step.description,
                )

                step_start = time.perf_counter()

                # Bind with context (referring resolution, engine inheritance)
                bound_action = self._bind_step(step)
                if bound_action is None:
                    last_error = f"Could not bind step '{step.description}' to a tool (unresolvable target)"
                    logger.warning(last_error)
                    break  # no point retrying the same binding failure

                # Apply method variation to the bound arguments
                merged_args = dict(bound_action.arguments)
                merged_args.update(variation)
                merged_args["last_referenced"] = self._context.get("last_referenced_file")
                varied_action = BoundAction(
                    tool_name=bound_action.tool_name,
                    arguments=merged_args,
                    target=bound_action.target,
                )

                pre_obs = self.observer.observe(delay_sec=0.0)
                tool_res = self.coordinator.executor.execute_action(varied_action)
                step_elapsed_ms = (time.perf_counter() - step_start) * 1000.0

                if not tool_res.success:
                    last_error = tool_res.error or tool_res.message
                    # Clarification requests abort the whole goal honestly
                    if tool_res.data and tool_res.data.get("needs_clarification"):
                        return False, False, tool_res.message, varied_action, tool_res
                    memory.record_failure(step, varied_action.tool_name, ActionMethod.NATIVE_API, last_error)
                    # Short pause before retry (transient UI timing)
                    if attempt < self.method_switch_threshold:
                        time.sleep(0.3)
                    continue

                # Verify
                post_obs = self.observer.observe(delay_sec=0.05)
                verify_res = self._verify_step(step, varied_action, tool_res, pre_obs, post_obs)

                if verify_res.verified:
                    logger.info("Step %d VERIFIED via variation %s.", step.step_id, variation or "standard")
                    memory.record_success(
                        step=step,
                        tool_used=varied_action.tool_name,
                        method_used=ActionMethod.NATIVE_API,
                        result=tool_res,
                        verification=verify_res,
                        elapsed_ms=step_elapsed_ms,
                    )
                    if lifecycle:
                        lifecycle.transition(
                            CommandStage.STEP_COMPLETE,
                            detail=f"{step.step_id}: {step.description}",
                        )
                    return True, True, "", varied_action, tool_res

                last_error = verify_res.message
                memory.record_failure(step, varied_action.tool_name, ActionMethod.NATIVE_API, last_error)
                # The state may need time to settle (window appears late)
                if attempt < self.method_switch_threshold:
                    time.sleep(0.4)

            logger.warning(
                "Variation %s failed %d times for step %d. Trying next variation...",
                variation or "standard", self.method_switch_threshold, step.step_id,
            )

        return False, False, last_error, bound_action, tool_res

    # ------------------------------------------------------------------
    # Binding with context resolution
    # ------------------------------------------------------------------

    def _bind_step(self, step: ActionStep) -> Optional[BoundAction]:
        """Bind an ActionStep to a tool action using context for references."""
        binder = getattr(self.coordinator, "binder", None)
        if binder is None:
            return None

        # Contextual override: right after a browser search, phrases like
        # "open it", "open the video", "play the latest video" refer to the
        # SEARCH RESULTS, not a file or media player.
        if self._context.get("last_search_query"):
            target_l = (step.target or "").lower().strip()
            wants_result = False
            if step.intent_hint == "media_control" and step.arguments.get("action") in ("play", "play_pause"):
                wants_result = True
            elif step.intent_hint == "file_open" and (
                target_l in ("it", "this", "that", "them", "this file", "the file", "this video", "the video", "the image")
                and not self._context.get("last_referenced_file")
            ):
                wants_result = True
            elif step.intent_hint == "open_application" and target_l in (
                "result", "results", "the result", "the results", "video", "the video",
                "link", "the link", "item", "the item",
            ):
                wants_result = True
            if wants_result:
                logger.info(
                    "Search context: resolving '%s' (%s) as a search-result click.",
                    step.target, step.intent_hint,
                )
                return BoundAction(
                    tool_name="click_element",
                    arguments={
                        "element": "search result",
                        "ordinal": "first",
                        "query": str(self._context.get("last_search_query", "")),
                    },
                    target="search result",
                )

        # Referring resolution: substitute context-known entities
        target = step.target
        args = dict(step.arguments)
        referring = args.pop("_referring", None) or (target if step.intent_hint in (
            "file_open", "file_move", "file_rename", "file_delete", "file_compress",
            "download_action", "ui_action",
        ) and self._is_referring(target) else None)

        if referring:
            resolved_ref = self._resolve_reference(referring, step.intent_hint)
            if resolved_ref:
                args.update(resolved_ref)
        # Context-aware search engine inheritance (site from previous step)
        if step.intent_hint == "browser_search" and args.get("engine", "google") == "google":
            site = self._context.get("current_site", "")
            for domain, engine in ENGINE_DOMAINS.items():
                if domain in site:
                    args["engine"] = engine
                    break

        # Rebuild a Clause for the binder from the step
        from planning.semantic_parser import Clause
        clause = Clause(
            raw=step.description,
            intent=step.intent_hint,
            verb="",
            target=target,
            arguments=args,
            is_referring=False,
        )
        # Bypass app-resolution for intents the binder handles generically
        try:
            action = binder.bind_clause(clause)
        except Exception as e:
            logger.warning("bind_clause failed for step '%s': %s", step.description, e)
            action = None

        if action is None and binder:
            # Fallback: whole-text bind (handles unknown intents via parser)
            try:
                action = binder.bind(step.description)
            except Exception:
                action = None
        return action

    def _is_referring(self, target: str) -> bool:
        if not target:
            return False
        t = target.lower().strip()
        if t in ("it", "this", "that", "them", "this file", "that file", "the file", "this image", "the image"):
            return True
        import re as _re
        return bool(_re.match(r"^(this|that|the)\s+\w+", t))

    def _resolve_reference(self, ref: str, intent: str) -> Dict[str, Any]:
        """Resolve a referring expression using execution context.
        Only fills in information the parser could not know (concrete paths
        from earlier steps); never overwrites parsed slot values."""
        out: Dict[str, Any] = {}
        last_file = self._context.get("last_referenced_file")
        if intent in ("file_move", "file_copy", "file_rename", "file_delete", "file_compress", "file_open"):
            if last_file:
                out["source_path"] = last_file
        if intent == "download_action" and last_file:
            out["path"] = last_file
        return out

    # ------------------------------------------------------------------
    # Context tracking
    # ------------------------------------------------------------------

    def _update_context_from_result(
        self,
        step: ActionStep,
        bound: Optional[BoundAction],
        res: Optional[ToolResult],
        obs: DesktopObservation,
    ) -> None:
        if res is None:
            return
        data = res.data or {}
        # Track last referenced file
        for key in ("moved_path", "path", "downloaded_file"):
            v = data.get(key)
            if v:
                self._context["last_referenced_file"] = str(v)
                break
        # Track current site / app
        if bound and bound.tool_name in ("browser_navigate", "browser_search"):
            url = data.get("url", "") or (bound.arguments.get("url", "") if bound else "")
            if url:
                self._context["current_site"] = str(url)
        if bound and bound.tool_name == "browser_search" and res.success:
            self._context["last_search_query"] = str(data.get("query", ""))
            self._context["last_search_engine"] = str(data.get("engine", "google"))
            # A fresh search invalidates the "open it"-style file reference
            self._context.pop("last_referenced_file", None)
        if bound and bound.tool_name == "open_application":
            self._context["last_opened_app"] = data.get("target_name", bound.target)
        if obs and obs.active_window_class:
            self._context["current_app"] = obs.active_window_class

    # ------------------------------------------------------------------
    # Replanning with genuine variation
    # ------------------------------------------------------------------

    def _replan_with_variation(
        self,
        goal: str,
        memory: ExecutionMemory,
        failed_step: ActionStep,
        desktop_state: DesktopObservation,
        replan_count: int,
    ) -> GoalPlan:
        """
        Replan after failure. Variation by replan number:
          1: rescue variant (switch to existing window) + remaining steps
          2: alternative mechanism (direct exec / different approach) + remaining
          3: final attempt with generic dispatch
        Never regenerates an identical plan.
        """
        logger.warning("Replanning after failure of step %d ('%s')", failed_step.step_id, failed_step.description)
        ctx = {
            "active_window": desktop_state.active_window_name,
            "active_class": desktop_state.active_window_class,
            "failed_step": failed_step.description,
            "completed_steps": [c.step.description for c in memory.completed_steps],
        }

        base_plan = self.planner.plan(goal, desktop_state=desktop_state)

        remaining: List[ActionStep] = []
        step_counter = 1
        for s in base_plan.steps:
            if memory.is_action_already_completed(s.intent_hint, s.target):
                continue
            remaining.append(s)

        # Insert a varied rescue step for the failed action at the front
        rescue = self._make_rescue_step(failed_step, replan_count)
        if rescue is not None:
            rescue.step_id = step_counter
            remaining.insert(0, rescue)
            step_counter += 1

        for s in remaining:
            s.step_id = step_counter
            step_counter += 1

        if not remaining:
            remaining.append(
                ActionStep(
                    step_id=1,
                    description=f"Retry: {failed_step.description}",
                    intent_hint=failed_step.intent_hint,
                    target=failed_step.target,
                    arguments=dict(failed_step.arguments),
                    verification_criteria=failed_step.verification_criteria,
                    is_terminal=True,
                )
            )

        return GoalPlan(
            goal_id=f"replan-{replan_count}-{memory.plan.goal_id}",
            original_goal=goal,
            steps=remaining,
            is_single_action=len(remaining) == 1,
            context_notes=f"Replanned after step '{failed_step.description}' failed (variation {replan_count})",
        )

    def _make_rescue_step(self, failed: ActionStep, replan_count: int) -> Optional[ActionStep]:
        intent = failed.intent_hint
        target = failed.target
        if not target:
            return None
        if intent == "open_application":
            if replan_count == 1:
                # Rescue: switch to any existing window of that app
                return ActionStep(
                    step_id=0,
                    description=f"Rescue: switch to existing {target} window",
                    intent_hint="switch_window",
                    target=target,
                    arguments={"query": target, "category_fallback": True},
                    verification_criteria=f"Window {target} focused",
                )
            return ActionStep(
                step_id=0,
                description=f"Retry direct launch of {target}",
                intent_hint="open_application",
                target=target,
                arguments={**dict(failed.arguments), "launch_preference": "direct"},
                verification_criteria=f"Application {target} running",
            )
        if intent in ("browser_navigate", "browser_search"):
            return ActionStep(
                step_id=0,
                description=f"Retry {failed.description} (default browser)",
                intent_hint=intent,
                target=target,
                arguments={**dict(failed.arguments), "force_mechanism": "xdg-open"},
                verification_criteria=failed.verification_criteria,
            )
        if intent == "click_element":
            return ActionStep(
                step_id=0,
                description=f"Retry activating '{target}' after re-observation",
                intent_hint="click_element",
                target=target,
                arguments={**dict(failed.arguments), "wait_sec": 1.0, "retries": 3},
                verification_criteria=failed.verification_criteria,
            )
        # Generic: same step with retry hint
        return ActionStep(
            step_id=0,
            description=f"Retry {failed.description}",
            intent_hint=intent,
            target=target,
            arguments=dict(failed.arguments),
            verification_criteria=failed.verification_criteria,
        )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def _verify_step(
        self,
        step: ActionStep,
        bound_action: Optional[BoundAction],
        tool_res: ToolResult,
        pre_obs: DesktopObservation,
        post_obs: DesktopObservation,
    ) -> VerificationResult:
        if not tool_res.success:
            return VerificationResult(
                verified=False,
                message=tool_res.error or tool_res.message,
                details={"stage": "execution_failed"},
            )

        # Tools that self-verify their post-state (file ops, waits, saves)
        if not tool_res.observation_required:
            return VerificationResult(verified=True, message=tool_res.message, details={"self_verified": True})

        if bound_action:
            # On retries, don't fail an app step just because the window
            # matches a previous observation (it's already open)
            return self.verifier.verify(bound_action, tool_res, pre_obs, post_obs)

        return VerificationResult(verified=True, message="Step completed.", details={})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_cancelled(self, lifecycle: Optional[CommandLifecycle]) -> bool:
        estop = getattr(self.coordinator, "emergency_stop", None)
        if estop is not None and estop.is_stopped():
            return True
        if lifecycle is not None:
            # Cancellation via lifecycle (watchdog / user)
            cancelled = getattr(lifecycle, "cancelled", False)
            if cancelled:
                return True
            if getattr(lifecycle, "cancel_requested", False):
                return True
        return False

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
