"""
Window Control Tools for Laya Desktop Agent.
Handles minimizing, maximizing, resizing, and desktop switching via kdotool.
"""

from typing import Optional
import logging
from tools.base import Tool, ToolResult, SafetyTier
from integrations.kdotool_client import KdotoolClient

logger = logging.getLogger(__name__)


class MinimizeWindowTool(Tool):
    name = "minimize_window"
    description = "Minimize the active or target window"
    safety_tier = SafetyTier.SAFE

    def __init__(self, kdotool: KdotoolClient):
        self.kdotool = kdotool

    def execute(self, window_id: Optional[str] = None, **kwargs) -> ToolResult:
        success = self.kdotool.minimize_window(window_id)
        return ToolResult(
            success=success,
            message="Window minimized." if success else "Failed to minimize window.",
            error=None if success else "kdotool minimize error",
            observation_required=True,
        )


class MaximizeWindowTool(Tool):
    name = "maximize_window"
    description = "Maximize the active or target window"
    safety_tier = SafetyTier.SAFE

    def __init__(self, kdotool: KdotoolClient):
        self.kdotool = kdotool

    def execute(self, window_id: Optional[str] = None, **kwargs) -> ToolResult:
        success = self.kdotool.maximize_window(window_id)
        return ToolResult(
            success=success,
            message="Window maximized." if success else "Failed to maximize window.",
            error=None if success else "kdotool maximize error",
            observation_required=True,
        )


class SwitchDesktopTool(Tool):
    name = "switch_desktop"
    description = "Switch virtual desktop number (1, 2, 3...)"
    safety_tier = SafetyTier.SAFE

    def __init__(self, kdotool: KdotoolClient):
        self.kdotool = kdotool

    def execute(self, number: int = 1, **kwargs) -> ToolResult:
        success = self.kdotool.set_desktop(number)
        return ToolResult(
            success=success,
            message=f"Switched to desktop {number}." if success else f"Failed to switch to desktop {number}.",
            data={"desktop": number},
            error=None if success else "kdotool set_desktop error",
            observation_required=True,
        )
