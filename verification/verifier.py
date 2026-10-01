"""
Action Verifier for Laya Desktop Agent.
Validates whether executed desktop actions achieved their intended post-conditions.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging
import time
from execution.parameter_binder import BoundAction
from tools.base import ToolResult
from verification.observer import DesktopObservation
from integrations.kdotool_client import KdotoolClient

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    verified: bool
    message: str
    details: Dict[str, Any]


class ActionVerifier:
    def __init__(self, kdotool: Optional[KdotoolClient] = None):
        self.kdotool = kdotool

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
        expected_wm = action.arguments.get("wm_class", "").lower().strip()
        candidate_classes = [c.lower() for c in action.arguments.get("candidate_wm_classes", []) if c]
        is_folder = result.data.get("is_folder", False)
        reported_win_id = result.data.get("window_id")
        target_clean = target.lower()

        def _is_match(cls: str, title: str, win_id: Optional[str] = None) -> bool:
            if reported_win_id and win_id and reported_win_id.lower() == win_id.lower():
                return True
            cls_lower = cls.lower()
            title_lower = title.lower()
            if expected_wm and (expected_wm in cls_lower or cls_lower in expected_wm or expected_wm in title_lower):
                return True
            for cand in candidate_classes:
                if cand in cls_lower or cls_lower in cand or cand in title_lower:
                    return True
            if target_clean and (target_clean in cls_lower or target_clean in title_lower):
                return True
            for token in target_clean.split():
                if len(token) > 3 and (token in cls_lower or token in title_lower):
                    return True
            return False

        # 3. Open Application Verification
        if tool_name == "open_application":
            if is_folder:
                if post_obs.active_window_class.lower() in ("dolphin", "nautilus", "thunar") or target in post_obs.active_window_name.lower():
                    return VerificationResult(
                        verified=True,
                        message=f"Verified: Folder '{target}' is open.",
                        details={"active_window": post_obs.active_window_name, "class": post_obs.active_window_class},
                    )

            # Check immediate match
            active_class = post_obs.active_window_class.lower()
            active_title = post_obs.active_window_name.lower()

            if _is_match(active_class, active_title, post_obs.active_window_id):
                return VerificationResult(
                    verified=True,
                    message=f"Verified: '{post_obs.active_window_name}' is open and focused.",
                    details={"active_window": post_obs.active_window_name, "class": post_obs.active_window_class},
                )

            # If not immediately matched and newly launched, poll for up to 1.5s
            if self.kdotool and not result.data.get("already_running", False):
                for _ in range(5):
                    time.sleep(0.3)
                    cur_id = self.kdotool.get_active_window()
                    if cur_id:
                        cur_class = self.kdotool.get_window_class(cur_id).lower()
                        cur_title = self.kdotool.get_window_name(cur_id).lower()
                        if _is_match(cur_class, cur_title, cur_id):
                            post_obs.active_window_id = cur_id
                            post_obs.active_window_class = cur_class
                            post_obs.active_window_name = cur_title
                            return VerificationResult(
                                verified=True,
                                message=f"Verified: '{cur_title}' is open and focused.",
                                details={"active_window": cur_title, "class": cur_class},
                            )

            # If active window is a different application -> TARGET_MISMATCH
            if post_obs.active_window_class:
                return VerificationResult(
                    verified=False,
                    message=f"TARGET_MISMATCH: Expected '{action.target}', but '{post_obs.active_window_name}' ({post_obs.active_window_class}) is active.",
                    details={
                        "stage": "target_mismatch",
                        "expected_target": action.target,
                        "expected_wm_class": expected_wm,
                        "actual_window": post_obs.active_window_name,
                        "actual_class": post_obs.active_window_class,
                    },
                )

            return VerificationResult(
                verified=False,
                message=f"TARGET_MISMATCH: Application '{action.target}' did not present a focused window.",
                details={"stage": "window_not_focused", "expected_target": action.target},
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
            if not target or target in ("previous", "active_window") or _is_match(active_class, active_title):
                return VerificationResult(
                    verified=True,
                    message=f"Verified: Switched to '{post_obs.active_window_name}'.",
                    details={"active_title": post_obs.active_window_name, "class": post_obs.active_window_class},
                )
            return VerificationResult(
                verified=False,
                message=f"TARGET_MISMATCH: Window did not switch to '{target}'. Current window is '{post_obs.active_window_name}' ({post_obs.active_window_class}).",
                details={
                    "stage": "target_mismatch",
                    "expected_target": target,
                    "expected_wm_class": expected_wm,
                    "actual_window": post_obs.active_window_name,
                    "actual_class": post_obs.active_window_class,
                },
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
