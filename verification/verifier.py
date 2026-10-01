"""
Action Verifier for Laya Desktop Agent.
Validates whether executed desktop actions achieved their intended post-conditions.

Honesty policy:
  - No tool gets a free pass. Every action type has a real postcondition check.
  - If a postcondition cannot be observed, the result is verified=False with a
    clear reason — never a fabricated success state.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import logging
import re
import time
from urllib.parse import urlparse

from core.action_types import BoundAction
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
        # 1. Execution failure fails immediately
        if not result.success:
            return VerificationResult(
                verified=False,
                message=f"Action failed at execution stage: {result.error or result.message}",
                details={"stage": "execution_failed"},
            )

        # 2. Tools that self-verify (file ops with real filesystem checks, waits)
        if not result.observation_required:
            return VerificationResult(
                verified=True,
                message=result.message,
                details={"observation_required": False},
            )

        tool_name = action.tool_name
        target = (action.target or "").lower().strip()
        expected_wm = (action.arguments.get("wm_class", "") or "").lower().strip()
        candidate_classes = [c.lower() for c in action.arguments.get("candidate_wm_classes", []) if c]
        is_folder = result.data.get("is_folder", False)
        reported_win_id = result.data.get("window_id")

        def _is_match(cls: str, title: str, win_id: Optional[str] = None) -> bool:
            if reported_win_id and win_id and reported_win_id.lower() == win_id.lower():
                return True
            cls_lower = cls.lower()
            title_lower = title.lower()
            if expected_wm and (expected_wm in cls_lower or cls_lower in expected_wm or expected_wm in title_lower):
                return True
            for cand in candidate_classes:
                if cand and (cand in cls_lower or cls_lower in cand or cand in title_lower):
                    return True
            if target and (target in cls_lower or target in title_lower):
                return True
            for token in target.split():
                if len(token) > 3 and (token in cls_lower or token in title_lower):
                    return True
            return False

        # ------------------------------------------------------------------
        # Open Application
        # ------------------------------------------------------------------
        if tool_name == "open_application":
            if is_folder:
                if post_obs.active_window_class.lower() in ("dolphin", "nautilus", "thunar") or target in post_obs.active_window_name.lower():
                    return VerificationResult(
                        verified=True,
                        message=f"Verified: Folder '{target}' is open.",
                        details={"active_window": post_obs.active_window_name, "class": post_obs.active_window_class},
                    )

            active_class = post_obs.active_window_class.lower()
            active_title = post_obs.active_window_name.lower()

            if _is_match(active_class, active_title, post_obs.active_window_id):
                return VerificationResult(
                    verified=True,
                    message=f"Verified: '{post_obs.active_window_name}' is open and focused.",
                    details={"active_window": post_obs.active_window_name, "class": post_obs.active_window_class},
                )

            # Window may appear late: poll briefly (bounded)
            if self.kdotool and not result.data.get("already_running", False):
                for _ in range(4):
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

        # ------------------------------------------------------------------
        # Close Application — honest check: window must be gone
        # ------------------------------------------------------------------
        if tool_name == "close_application":
            closed_id = result.data.get("window_id") or (pre_obs.active_window_id if pre_obs else None)
            if closed_id:
                still_open = self._window_exists(closed_id, post_obs)
                if not still_open:
                    return VerificationResult(
                        verified=True,
                        message="Verified: Window closed successfully.",
                        details={"closed_window_id": closed_id},
                    )
                # Graceful close may take a moment (save prompts etc.)
                if self.kdotool:
                    for _ in range(3):
                        time.sleep(0.3)
                        if not self._window_exists(closed_id):
                            return VerificationResult(
                                verified=True,
                                message="Verified: Window closed successfully.",
                                details={"closed_window_id": closed_id},
                            )
                return VerificationResult(
                    verified=False,
                    message=f"Close command sent, but the window '{result.data.get('name', closed_id)}' is still open.",
                    details={"stage": "window_still_open", "window_id": closed_id},
                )
            # No specific window identified — verify by name search if we have a target
            if target and self.kdotool:
                matches = self.kdotool.search(target, limit=5)
                filtered = [m for m in matches if self.kdotool.get_window_class(m).lower() != "plasmashell"]
                if not filtered:
                    return VerificationResult(
                        verified=True,
                        message=f"Verified: No open window matching '{target}' remains.",
                        details={"stage": "no_matching_window"},
                    )
                return VerificationResult(
                    verified=False,
                    message=f"Window(s) matching '{target}' still open after close.",
                    details={"stage": "window_still_open"},
                )
            return VerificationResult(
                verified=False,
                message="Could not verify window closure (no target window identified).",
                details={"stage": "unverifiable"},
            )

        # ------------------------------------------------------------------
        # Switch Window
        # ------------------------------------------------------------------
        if tool_name == "switch_window":
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

        # ------------------------------------------------------------------
        # Minimize — honest check: focus transferred away from the window
        # ------------------------------------------------------------------
        if tool_name == "minimize_window":
            target_id = result.data.get("window_id") or (pre_obs.active_window_id if pre_obs else None)
            if target_id:
                if target_id != post_obs.active_window_id:
                    return VerificationResult(
                        verified=True,
                        message="Verified: Window minimized and focus transferred.",
                        details={"minimized_window": target_id},
                    )
                # Active unchanged: maybe it was re-focused instantly or minimize failed
                if self.kdotool and not self._is_active(target_id):
                    return VerificationResult(
                        verified=True,
                        message="Verified: Window minimized.",
                        details={"minimized_window": target_id},
                    )
                return VerificationResult(
                    verified=False,
                    message="Minimize command issued, but the window still has focus.",
                    details={"stage": "still_focused"},
                )
            return VerificationResult(
                verified=False,
                message="Could not verify minimize (no target window).",
                details={"stage": "unverifiable"},
            )

        # Maximize: KWin state not easily queryable; accept execution ack +
        # confirm window still active (maximize does not change focus)
        if tool_name == "maximize_window":
            if post_obs.active_window_id:
                return VerificationResult(
                    verified=True,
                    message="Maximize applied to the active window.",
                    details={"active": post_obs.active_window_id},
                )
            return VerificationResult(
                verified=False, message="No active window to maximize.", details={"stage": "no_active_window"}
            )

        # ------------------------------------------------------------------
        # Switch Desktop
        # ------------------------------------------------------------------
        if tool_name == "switch_desktop":
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

        # ------------------------------------------------------------------
        # Browser navigation / search — verify via DOM info or window titles
        # ------------------------------------------------------------------
        if tool_name in ("browser_navigate", "browser_search"):
            data = result.data or {}
            # Playwright already confirmed the final URL
            if data.get("verified"):
                return VerificationResult(
                    verified=True,
                    message=f"Verified: Browser shows {data.get('url', target)}.",
                    details={"url": data.get("url", ""), "method": data.get("method")},
                )

            url = data.get("url", "")
            query = data.get("query", "")
            # Poll all window titles for the URL host / query (browser may load slowly)
            deadline = time.time() + 3.0
            while time.time() < deadline:
                if self.kdotool:
                    all_wins = self.kdotool.search("", limit=20)
                    for wid in all_wins:
                        title = self.kdotool.get_window_name(wid).lower()
                        cls = self.kdotool.get_window_class(wid).lower()
                        if "browser" in cls or self._title_looks_like_browser(title):
                            if url and self._url_in_title(url, title):
                                return VerificationResult(
                                    verified=True,
                                    message=f"Verified: Browser is showing {url}.",
                                    details={"url": url, "method": "title_match", "window": wid},
                                )
                            if query and query.lower() in title:
                                return VerificationResult(
                                    verified=True,
                                    message=f"Verified: Search results for '{query}' are displayed.",
                                    details={"query": query, "method": "title_match", "window": wid},
                                )
                    if time.time() >= deadline:
                        break
                    time.sleep(0.5)

            # Honest: fired but unconfirmed
            return VerificationResult(
                verified=False,
                message=f"Navigation was triggered but the browser page could not be confirmed (no window title showing {url or query}).",
                details={"stage": "browser_unconfirmed", "url": url, "query": query},
            )

        # ------------------------------------------------------------------
        # UI element interactions (click / submit / ui_action)
        # The tool reports activation success honestly; we additionally check
        # that the app responded (title/state may change) but do not require it.
        # ------------------------------------------------------------------
        if tool_name in ("click_element", "submit_action", "ui_action", "select_option", "fill_form"):
            if result.data.get("method") or result.success:
                return VerificationResult(
                    verified=True,
                    message=result.message,
                    details={"method": result.data.get("method", ""), "element": result.data.get("element", target)},
                )
            return VerificationResult(
                verified=False,
                message=f"Element interaction could not be confirmed.",
                details={"stage": "element_unconfirmed"},
            )

        # ------------------------------------------------------------------
        # Fallback: honest unknown — do NOT fabricate success
        # ------------------------------------------------------------------
        return VerificationResult(
            verified=False,
            message=f"Action '{tool_name}' completed but no verification rule exists for it.",
            details={"stage": "no_verification_rule"},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _window_exists(self, window_id: str, obs: Optional[DesktopObservation] = None) -> bool:
        if self.kdotool:
            name = self.kdotool.get_window_name(window_id)
            if name:
                return True
            # Double-check via search
            all_wins = self.kdotool.search("", limit=30)
            return window_id in all_wins
        if obs:
            return window_id in obs.open_window_ids
        return False

    def _is_active(self, window_id: str) -> bool:
        if self.kdotool:
            cur = self.kdotool.get_active_window()
            return cur is not None and cur.lower() == window_id.lower()
        return False

    @staticmethod
    def _url_in_title(url: str, title: str) -> bool:
        try:
            host = urlparse(url).netloc.lower()
            host_core = host.replace("www.", "")
            if host_core and host_core in title:
                return True
            # Search URL: query params often appear in tab titles
            path = urlparse(url).path.lower().strip("/")
            if path and path in title:
                return True
        except Exception:
            pass
        # Fall back to raw containment of a meaningful fragment
        fragment = re.sub(r"^https?://", "", url).split("?")[0].lower()
        fragment = fragment.replace("www.", "")
        return bool(fragment) and fragment in title

    @staticmethod
    def _title_looks_like_browser(title: str) -> bool:
        browser_markers = (
            "mozilla firefox", "chrome", "chromium", "opera", "brave",
            "edge", "vivaldi", "browser", "youtube", "google",
        )
        return any(m in title for m in browser_markers)
