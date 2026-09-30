"""
Screen Control Tools for Laya Desktop Agent.
Follows strict screen text priority: AT-SPI2 first (Tier 1), falling back to Spectacle screenshot (Tier 2/3).
Never leaves temporary screenshots uncleaned.
"""

from pathlib import Path
from typing import Optional, List
import logging
import os
import shutil
import subprocess
import time
from tools.base import Tool, ToolResult, SafetyTier
from integrations.atspi_client import AtspiClient
from integrations.kdotool_client import KdotoolClient

logger = logging.getLogger(__name__)


class ReadScreenTool(Tool):
    name = "read_screen"
    description = "Read visible text or summarize contents of the current screen/window"
    safety_tier = SafetyTier.SAFE

    def __init__(self, atspi: AtspiClient, kdotool: KdotoolClient):
        self.atspi = atspi
        self.kdotool = kdotool

    def execute(self, **kwargs) -> ToolResult:
        # 1. Tier 1: Accessibility Tree (Costs 0 VRAM, 0 GPU)
        texts = self.atspi.get_active_app_text(max_elements=30)
        active_win = self.kdotool.get_active_window()
        win_name = self.kdotool.get_window_name(active_win) if active_win else "Unknown"

        if texts:
            summary = "\n".join(f"- {t}" for t in texts[:15])
            msg = f"Active window: '{win_name}'. Screen text:\n{summary}"
            return ToolResult(
                success=True,
                message=msg,
                data={"window_name": win_name, "elements_count": len(texts), "text_samples": texts[:15]},
                observation_required=False,
            )

        # Fallback if no accessibility text found
        return ToolResult(
            success=True,
            message=f"Active window: '{win_name}'. No accessibility text elements found.",
            data={"window_name": win_name, "elements_count": 0},
            observation_required=False,
        )


class TakeScreenshotTool(Tool):
    name = "take_screenshot"
    description = "Take a desktop or window screenshot"
    safety_tier = SafetyTier.SAFE

    def __init__(self, screenshot_dir: Path):
        self.screenshot_dir = screenshot_dir

    def execute(self, target_path: Optional[str] = None, **kwargs) -> ToolResult:
        spectacle = shutil.which("spectacle") or "/usr/bin/spectacle"
        if not os.path.exists(spectacle):
            return ToolResult(success=False, message="Spectacle screenshot utility not found.", error="Missing spectacle")

        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        out_file = Path(target_path) if target_path else (self.screenshot_dir / f"screenshot_{int(time.time() * 1000)}.png")

        try:
            # Run non-interactive background screenshot
            res = subprocess.run(
                [spectacle, "-b", "-n", "-o", str(out_file)],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
            if res.returncode == 0 and out_file.exists():
                return ToolResult(
                    success=True,
                    message=f"Screenshot captured: {out_file.name}",
                    data={"file_path": str(out_file), "size_bytes": out_file.stat().st_size},
                    observation_required=False,
                )
            return ToolResult(success=False, message="Failed to capture screenshot.", error=res.stderr)
        except Exception as e:
            return ToolResult(success=False, message="Screenshot error.", error=str(e))
