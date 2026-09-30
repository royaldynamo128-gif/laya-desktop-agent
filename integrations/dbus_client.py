"""
KDE DBus Client for Laya Desktop Agent.
Handles system notifications, media playback control (MPRIS), and Plasma interactions.
"""

from typing import Optional, List
import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)


class DBusClient:
    def __init__(self):
        self.qdbus_tool = shutil.which("qdbus6") or shutil.which("qdbus") or shutil.which("busctl")

    def show_notification(self, title: str, message: str, app_name: str = "Laya Agent", timeout_ms: int = 3000) -> bool:
        """Show desktop notification via notify-send or DBus."""
        notify_send = shutil.which("notify-send")
        if notify_send:
            try:
                subprocess.run(
                    [notify_send, "-a", app_name, "-t", str(timeout_ms), title, message],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except Exception as e:
                logger.debug("notify-send failed: %s", e)

        # Fallback to qdbus6
        if self.qdbus_tool and "qdbus" in self.qdbus_tool:
            try:
                cmd = [
                    self.qdbus_tool,
                    "org.freedesktop.Notifications",
                    "/org/freedesktop/Notifications",
                    "org.freedesktop.Notifications.Notify",
                    app_name,
                    "0",
                    "",
                    title,
                    message,
                    "[]",
                    "{}",
                    str(timeout_ms),
                ]
                subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception as e:
                logger.debug("DBus notify failed: %s", e)

        return False

    def media_play_pause(self) -> bool:
        """Toggle media play/pause on active MPRIS player."""
        playerctl = shutil.which("playerctl")
        if playerctl:
            try:
                res = subprocess.run([playerctl, "play-pause"], check=False)
                return res.returncode == 0
            except Exception:
                pass

        if self.qdbus_tool and "qdbus" in self.qdbus_tool:
            try:
                # Find active players
                proc = subprocess.run([self.qdbus_tool], capture_output=True, text=True, check=False)
                players = [line.strip() for line in proc.stdout.splitlines() if "org.mpris.MediaPlayer2" in line]
                for p in players:
                    subprocess.run(
                        [self.qdbus_tool, p, "/org/mpris/MediaPlayer2", "org.mpris.MediaPlayer2.Player.PlayPause"],
                        check=False,
                        stdout=subprocess.DEVNULL,
                    )
                return len(players) > 0
            except Exception as e:
                logger.debug("DBus MPRIS play-pause failed: %s", e)

        return False
