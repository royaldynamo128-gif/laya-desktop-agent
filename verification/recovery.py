"""
Recovery and Retry Manager for Laya Desktop Agent.
Executes bounded alternative strategies when action verification fails.
"""

from typing import Optional, Callable, Tuple
import logging
import time
from execution.parameter_binder import BoundAction
from tools.base import ToolResult
from verification.verifier import VerificationResult

logger = logging.getLogger(__name__)


class RecoveryManager:
    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    def attempt_recovery(
        self,
        action: BoundAction,
        verification: VerificationResult,
        execute_fn: Callable[[BoundAction], ToolResult],
        verify_fn: Callable[[BoundAction, ToolResult], VerificationResult],
    ) -> Tuple[Optional[ToolResult], VerificationResult]:
        """Attempt alternative execution paths if initial verification failed."""
        if verification.verified:
            return None, verification

        logger.warning("Initiating recovery for action [%s]: %s", action.tool_name, verification.message)

        for attempt in range(1, self.max_retries + 1):
            logger.info("Recovery attempt %d/%d for [%s]...", attempt, self.max_retries, action.tool_name)
            time.sleep(0.3)

            # Strategy 1: For open_application:
            # First attempt: Window rescue - check if window is already open or unminimized
            # Second attempt: Retry launching once (fallback launcher path)
            if action.tool_name == "open_application":
                wm_cls = action.arguments.get("wm_class", "")
                cand_classes = action.arguments.get("candidate_wm_classes", [])
                if attempt == 1:
                    logger.info("Recovery step 1: attempting window rescue via switch_window for '%s'...", action.target)
                    alt_action = BoundAction(
                        tool_name="switch_window",
                        arguments={
                            "query": action.target,
                            "wm_class": wm_cls,
                            "candidate_wm_classes": cand_classes,
                        },
                        target=action.target,
                    )
                    alt_result = execute_fn(alt_action)
                    alt_verify = verify_fn(alt_action, alt_result)
                    if alt_verify.verified:
                        logger.info("Recovery successful on attempt %d via window rescue!", attempt)
                        return alt_result, alt_verify
                elif attempt == 2:
                    logger.info("Recovery step 2: attempting single relaunch for '%s'...", action.target)
                    alt_result = execute_fn(action)
                    alt_verify = verify_fn(action, alt_result)
                    if alt_verify.verified:
                        logger.info("Recovery successful on attempt %d via relaunch!", attempt)
                        return alt_result, alt_verify
                else:
                    # Bounded: Never launch a 3rd instance
                    break

            # Strategy 2: For switch_window, try fuzzy substring or window cycling
            elif action.tool_name == "switch_window":
                alt_result = execute_fn(action)
                alt_verify = verify_fn(action, alt_result)
                if alt_verify.verified:
                    logger.info("Recovery successful on attempt %d!", attempt)
                    return alt_result, alt_verify

            # Strategy 3: Generic retry
            else:
                alt_result = execute_fn(action)
                alt_verify = verify_fn(action, alt_result)
                if alt_verify.verified:
                    logger.info("Recovery successful on attempt %d!", attempt)
                    return alt_result, alt_verify

        logger.error("All recovery attempts exhausted for [%s].", action.tool_name)
        return None, verification


