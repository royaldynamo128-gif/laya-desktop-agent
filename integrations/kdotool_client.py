"""
kdotool Client for KDE Plasma 6 Wayland window management.
Wraps /usr/local/bin/kdotool for window search, activation, minimization, closing, and desktop switching.
"""

from typing import List, Optional, Dict
import logging
import os
import re
import shutil
import subprocess

logger = logging.getLogger(__name__)


class KdotoolClient:
    def __init__(self, binary_path: Optional[str] = None):
        # Resolution order: explicit arg > $KDTOOL_BIN env > PATH > /usr/local/bin/kdotool
        self.binary_path = (
            binary_path
            or os.environ.get("KDTOOL_BIN")
            or shutil.which("kdotool")
            or "/usr/local/bin/kdotool"
        )

    def _run(self, *args) -> subprocess.CompletedProcess:
        cmd = [self.binary_path] + list(args)
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
        except Exception as e:
            logger.error("kdotool execution error for %s: %s", cmd, e)
            raise

    def get_active_window(self) -> Optional[str]:
        """Return the window ID of the currently active/focused window."""
        res = self._run("getactivewindow")
        if res.returncode == 0:
            out = res.stdout.strip()
            # Window IDs are formatted like {ce827a5d-60d2-4543-bbd0-5d51f99fbfee}
            match = re.search(r"\{[0-9a-fA-F\-]{36}\}", out)
            if match:
                return match.group(0)
            return out if out else None
        return None

    def get_window_name(self, window_id: str) -> str:
        """Get the title string displayed in window titlebar."""
        res = self._run("getwindowname", window_id)
        return res.stdout.strip() if res.returncode == 0 else ""

    def get_window_class(self, window_id: str) -> str:
        """Get the window class (e.g. google-chrome, opera, ghostty)."""
        res = self._run("getwindowclassname", window_id)
        return res.stdout.strip() if res.returncode == 0 else ""

    def search(self, pattern: str, limit: int = 5) -> List[str]:
        """Search windows matching pattern in title or class name."""
        args = ["search", "--limit", str(limit), pattern]
        res = self._run(*args)
        if res.returncode == 0:
            matches = re.findall(r"\{[0-9a-fA-F\-]{36}\}", res.stdout)
            return matches
        return []

    def get_desktop_for_window(self, window_id: str) -> Optional[int]:
        """Get the virtual desktop number for a window."""
        res = self._run("get_desktop_for_window", window_id)
        if res.returncode == 0:
            out = res.stdout.strip()
            if out.isdigit():
                return int(out)
        return None

    def unminimize_window(self, window_id: str) -> bool:
        """Remove minimized state from a window so it becomes visible."""
        res = self._run("windowstate", "--remove", "MINIMIZED", window_id)
        return res.returncode == 0

    def activate_window(self, window_id: str) -> bool:
        """Activate and bring window to foreground, unminimizing and switching workspace if needed."""
        # 1. Unminimize if minimized
        self.unminimize_window(window_id)
        
        # 2. Check virtual desktop and switch if needed
        win_desktop = self.get_desktop_for_window(window_id)
        if win_desktop is not None and win_desktop > 0:
            cur_desktop = self.get_desktop()
            if win_desktop != cur_desktop:
                self.set_desktop(win_desktop)

        # 3. Raise to top of window stack
        self._run("windowraise", window_id)

        # 4. Activate / focus
        res = self._run("windowactivate", window_id)
        return res.returncode == 0

    def search_any(self, patterns: List[str], limit: int = 5, exclude_shell: bool = True) -> List[str]:
        """Search windows matching any pattern in patterns, filtering out plasmashell if requested."""
        seen = []
        for p in patterns:
            if not p or not p.strip():
                continue
            matches = self.search(p.strip(), limit=limit)
            for m in matches:
                if m not in seen:
                    if exclude_shell:
                        cls_name = self.get_window_class(m).lower()
                        if cls_name == "plasmashell":
                            continue
                    seen.append(m)
                    if len(seen) >= limit:
                        return seen
        return seen

    def is_active_window(self, window_id: str) -> bool:
        """Check if the given window ID is currently the focused active window."""
        cur = self.get_active_window()
        return cur is not None and cur.lower() == window_id.lower()

    def close_window(self, window_id: str) -> bool:
        """Gracefully close a window."""
        res = self._run("windowclose", window_id)
        return res.returncode == 0

    def minimize_window(self, window_id: Optional[str] = None) -> bool:
        """Minimize window (defaults to active window)."""
        target = window_id or self.get_active_window()
        if not target:
            return False
        res = self._run("windowminimize", target)
        return res.returncode == 0

    def maximize_window(self, window_id: Optional[str] = None) -> bool:
        """Maximize window (defaults to active window)."""
        target = window_id or self.get_active_window()
        if not target:
            return False
        res = self._run("windowstate", "--add", "MAXIMIZED", target)
        return res.returncode == 0

    def resize_window(self, width: int, height: int, window_id: Optional[str] = None) -> bool:
        """Resize window."""
        target = window_id or self.get_active_window()
        if not target:
            return False
        res = self._run("windowsize", target, str(width), str(height))
        return res.returncode == 0

    def set_desktop(self, desktop_num: int) -> bool:
        """Switch virtual desktop."""
        res = self._run("set_desktop", str(desktop_num))
        return res.returncode == 0

    def get_desktop(self) -> int:
        """Get current virtual desktop number."""
        res = self._run("get_desktop")
        try:
            return int(res.stdout.strip())
        except ValueError:
            return 1
