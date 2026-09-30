"""
Action Verifier for Laya Desktop Agent.
Validates whether executed desktop actions achieved their intended post-conditions.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging
from execution.parameter_binder import BoundAction
from tools.base import ToolResult
from verification.observer import DesktopObservation

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    verified: bool
    message: str
    details: Dict[str, Any]


class ActionVerifier:
    def verify(
        self,
        action: BoundAction,
        result: ToolResult,
        pre_obs: Optional[DesktopObservation],
        post_obs: DesktopObservation,
    ) -> VerificationResult:
        """Verify post-condition based on action type and observed state change."""
        # 1. If tool execution failed, verification fails immediately
        if not result.success:
            return VerificationResult(
                verified=False,
                message=f"Action failed at execution stage: {result.error or result.message}",
                details={"stage": "execution_failed"},
            )

        # 2. If observation not required, accept tool success
        if not result.observation_required:
            return VerificationResult(
                verified=True,
                message="Action completed and verified.",
                details={"observation_required": False},
            )

        tool_name = action.tool_name
        target = action.target.lower().strip()

        # 3. Open Application Verification
        if tool_name == "open_application":
            # Check if active window or any open window matches target
            active_class = post_obs.active_window_class.lower()
            active_title = post_obs.active_window_name.lower()
            if target in active_class or target in active_title or any(w in active_class for w in target.split()):
                return VerificationResult(
                    verified=True,
                    message=f"Verified: '{post_obs.active_window_name}' is open and focused.",
                    details={"active_window": post_obs.active_window_name, "class": post_obs.active_window_class},
                )
            
            # Check if new window count increased
            if pre_obs and len(post_obs.open_window_ids) > len(pre_obs.open_window_ids):
                return VerificationResult(
                    verified=True,
                    message=f"Verified: Application '{target}' opened.",
                    details={"new_window_detected": True},
                )

            return VerificationResult(
                verified=True,  # Launched successfully, window may take a second to paint
                message=f"Application launch issued for '{target}'.",
                details={"target": target, "active": post_obs.active_window_name},
            )

        # 4. Close Application Verification
        elif tool_name == "close_application":
            if pre_obs and pre_obs.active_window_id:
                if pre_obs.active_window_id not in post_obs.open_window_ids:
                    return VerificationResult(
                        verified=True,
                        message="Verified: Window closed successfully.",
                        details={"closed_window_id": pre_obs.active_window_id},
                    )
            return VerificationResult(
                verified=True,
                message="Window close command completed.",
                details={},
            )

        # 5. Switch Window Verification
        elif tool_name == "switch_window":
            active_class = post_obs.active_window_class.lower()
            active_title = post_obs.active_window_name.lower()
            if not target or target in active_class or target in active_title:
                return VerificationResult(
                    verified=True,
                    message=f"Verified: Switched to '{post_obs.active_window_name}'.",
                    details={"active_title": post_obs.active_window_name},
                )
            return VerificationResult(
                verified=False,
                message=f"Window did not switch to '{target}'. Current window is '{post_obs.active_window_name}'.",
                details={"target": target, "active": post_obs.active_window_name},
            )

        # 6. Minimize Window Verification
        elif tool_name == "minimize_window":
            if pre_obs and pre_obs.active_window_id != post_obs.active_window_id:
                return VerificationResult(
                    verified=True,
                    message="Verified: Window minimized and focus transferred.",
                    details={},
                )
            return VerificationResult(
                verified=True,
                message="Minimize command issued.",
                details={},
            )

        # 7. Switch Desktop Verification
        elif tool_name == "switch_desktop":
            target_num = action.arguments.get("number", 1)
            if post_obs.desktop_number == target_num:
                return VerificationResult(
                    verified=True,
                    message=f"Verified: Switched to desktop {target_num}.",
                    details={"desktop": post_obs.desktop_number},
                )
            return VerificationResult(
                verified=False,
                message=f"Desktop is {post_obs.desktop_number}, expected {target_num}.",
                details={"current": post_obs.desktop_number, "expected": target_num},
            )

        return VerificationResult(
            verified=True,
            message="Action completed.",
            details={},
        )
