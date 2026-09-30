"""
System Control Tools for Laya Desktop Agent.
Handles volume, mute, and media playback control via pactl and DBus MPRIS.
"""

from typing import Optional, Union
import logging
import re
import shutil
import subprocess
from tools.base import Tool, ToolResult, SafetyTier
from integrations.dbus_client import DBusClient

logger = logging.getLogger(__name__)


class SetVolumeTool(Tool):
    name = "set_volume"
    description = "Set or adjust audio volume (e.g. 'set volume to 50%', 'volume up', 'volume down')"
    safety_tier = SafetyTier.SAFE

    def execute(self, level: Union[int, str] = "50%", **kwargs) -> ToolResult:
        pactl = shutil.which("pactl") or "/usr/bin/pactl"
        val = str(level).strip()

        # Parse relative vs absolute
        if "up" in val.lower():
            target = "+10%"
        elif "down" in val.lower():
            target = "-10%"
        else:
            match = re.search(r"\d+", val)
            if match:
                num = int(match.group(0))
                num = max(0, min(100, num))
                target = f"{num}%"
            else:
                target = "50%"

        try:
            res = subprocess.run(
                [pactl, "set-sink-volume", "@DEFAULT_SINK@", target],
                check=False,
                capture_output=True,
                text=True,
            )
            if res.returncode == 0:
                return ToolResult(
                    success=True,
                    message=f"Volume adjusted to {target}.",
                    data={"target_volume": target},
                    observation_required=False,
                )
            return ToolResult(success=False, message="Failed to adjust volume.", error=res.stderr)
        except Exception as e:
            return ToolResult(success=False, message="Volume adjustment error.", error=str(e))


class MuteVolumeTool(Tool):
    name = "mute_volume"
    description = "Mute or unmute desktop audio"
    safety_tier = SafetyTier.SAFE

    def execute(self, state: str = "toggle", **kwargs) -> ToolResult:
        pactl = shutil.which("pactl") or "/usr/bin/pactl"
        val = "toggle"
        if "unmute" in state.lower():
            val = "0"
        elif "mute" in state.lower():
            val = "1"

        try:
            res = subprocess.run([pactl, "set-sink-mute", "@DEFAULT_SINK@", val], check=False)
            return ToolResult(
                success=res.returncode == 0,
                message="Audio mute toggled.",
                data={"state": val},
                observation_required=False,
            )
        except Exception as e:
            return ToolResult(success=False, message="Mute error.", error=str(e))


class MediaControlTool(Tool):
    name = "media_control"
    description = "Control media playback (play, pause, next, previous)"
    safety_tier = SafetyTier.SAFE

    def __init__(self, dbus_client: DBusClient):
        self.dbus = dbus_client

    def execute(self, action: str = "play_pause", **kwargs) -> ToolResult:
        success = self.dbus.media_play_pause()
        return ToolResult(
            success=success,
            message="Media playback toggled." if success else "No active media players found.",
            data={"action": action},
            observation_required=False,
        )
