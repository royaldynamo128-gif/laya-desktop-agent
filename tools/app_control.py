"""
Application Control Tools for Laya Desktop Agent.
Handles opening, closing, switching, and focusing desktop applications and directory folders.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import os
import re
import shutil
import subprocess
from tools.base import Tool, ToolResult, SafetyTier
from integrations.flatpak_helper import AppLauncher
from integrations.kdotool_client import KdotoolClient

logger = logging.getLogger(__name__)

FOLDER_MAP: Dict[str, Path] = {
    "projects": Path.home() / "Projects",
    "project": Path.home() / "Projects",
    "downloads": Path.home() / "Downloads",
    "download": Path.home() / "Downloads",
    "documents": Path.home() / "Documents",
    "document": Path.home() / "Documents",
    "desktop": Path.home() / "Desktop",
    "home": Path.home(),
    "pictures": Path.home() / "Pictures",
    "picture": Path.home() / "Pictures",
    "music": Path.home() / "Music",
    "videos": Path.home() / "Videos",
    "video": Path.home() / "Videos",
}

CATEGORY_WINDOW_CLASSES: Dict[str, List[str]] = {
    "browser": ["google-chrome", "chrome", "opera", "brave", "firefox", "chromium"],
    "web": ["google-chrome", "chrome", "opera", "brave", "firefox", "chromium"],
    "terminal": ["ghostty", "konsole", "alacritty", "kitty", "foot", "terminal"],
    "console": ["ghostty", "konsole", "alacritty", "kitty", "foot", "terminal"],
    "files": ["dolphin", "nautilus", "thunar", "file manager"],
    "file manager": ["dolphin", "nautilus", "thunar"],
    "notes": ["obsidian"],
    "editor": ["code", "cursor", "nvim", "kate"],
    "music": ["spotify", "vlc", "clementine"],
    "chat": ["telegram", "discord", "slack"],
}


class OpenApplicationTool(Tool):
    name = "open_application"
    description = "Launch or open a desktop application or directory folder (e.g. Chrome, Telegram, Projects folder, Downloads)"
    safety_tier = SafetyTier.SAFE

    def __init__(self, launcher: AppLauncher, kdotool: KdotoolClient):
        self.launcher = launcher
        self.kdotool = kdotool

    def _detect_folder(self, name: str) -> Optional[Path]:
        clean_name = name.lower().strip()
        folder_clean = re.sub(r"^(my|the)\s+", "", clean_name)
        folder_clean = re.sub(r"\s+(folder|directory)$", "", folder_clean).strip()

        if folder_clean in FOLDER_MAP:
            return FOLDER_MAP[folder_clean]
        if (Path.home() / folder_clean).is_dir():
            return Path.home() / folder_clean
        if Path(name).is_dir():
            return Path(name)
        if Path(folder_clean).is_dir():
            return Path(folder_clean)
        return None

    def execute(self, name: str, **kwargs) -> ToolResult:
        if not name:
            return ToolResult(success=False, message="Application or folder name required.", error="Missing name parameter")

        # 1. Check if target is a known directory/folder
        target_folder = self._detect_folder(name)
        if target_folder:
            if not target_folder.exists():
                target_folder.mkdir(parents=True, exist_ok=True)
            opener = shutil.which("dolphin") or shutil.which("xdg-open")
            if opener:
                try:
                    subprocess.Popen(
                        [opener, str(target_folder)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    return ToolResult(
                        success=True,
                        message=f"Opened folder '{target_folder.name or str(target_folder)}'.",
                        data={"path": str(target_folder), "is_folder": True},
                        observation_required=True,
                    )
                except Exception as e:
                    logger.error("Failed to open folder %s: %s", target_folder, e)
                    return ToolResult(success=False, message=f"Failed to open folder: {e}", error=str(e))

        # 2. Check if application is already running, switch to it if so
        windows = self.kdotool.search(name, limit=1)
        if windows:
            self.kdotool.activate_window(windows[0])
            title = self.kdotool.get_window_name(windows[0])
            return ToolResult(
                success=True,
                message=f"Switched to running {name}.",
                data={"window_id": windows[0], "title": title, "already_running": True},
                observation_required=True,
            )

        # 3. Launch via AppLauncher
        success, msg = self.launcher.launch(name)
        return ToolResult(
            success=success,
            message=msg,
            data={"target_name": name, "already_running": False},
            error=None if success else msg,
            observation_required=True,
        )


class CloseApplicationTool(Tool):
    name = "close_application"
    description = "Close or quit a window or application"
    safety_tier = SafetyTier.SAFE

    def __init__(self, kdotool: KdotoolClient):
        self.kdotool = kdotool

    def execute(self, name: Optional[str] = None, window_id: Optional[str] = None, **kwargs) -> ToolResult:
        target_win = window_id
        if not target_win:
            if name:
                matches = self.kdotool.search(name, limit=1)
                if matches:
                    target_win = matches[0]
            else:
                target_win = self.kdotool.get_active_window()

        if not target_win:
            return ToolResult(success=False, message="No active or matching window found to close.", error="No target window")

        win_name = self.kdotool.get_window_name(target_win)
        success = self.kdotool.close_window(target_win)
        return ToolResult(
            success=success,
            message=f"Closed window '{win_name}'." if success else f"Failed to close window '{win_name}'.",
            data={"window_id": target_win, "name": win_name},
            error=None if success else "kdotool windowclose returned error",
            observation_required=True,
        )


class SwitchWindowTool(Tool):
    name = "switch_window"
    description = "Switch to, focus, or bring to front an open window by name, class, or keyword (e.g. Chrome, browser, terminal)"
    safety_tier = SafetyTier.SAFE

    def __init__(self, kdotool: KdotoolClient):
        self.kdotool = kdotool

    def execute(self, query: str = "", **kwargs) -> ToolResult:
        if not query:
            # Switch to previous window
            all_wins = self.kdotool.search("", limit=2)
            if len(all_wins) >= 2:
                target = all_wins[1]
                self.kdotool.activate_window(target)
                name = self.kdotool.get_window_name(target)
                return ToolResult(
                    success=True,
                    message=f"Switched to previous window '{name}'.",
                    data={"window_id": target, "name": name},
                    observation_required=True,
                )
            return ToolResult(success=False, message="No other windows available.", error="Stack empty")

        clean_query = re.sub(r"^(the|my|a|an)\s+", "", query.lower().strip())

        # 1. Direct search by query
        matches = self.kdotool.search(clean_query, limit=1)

        # 2. Check category mapping if direct search yields no matches
        if not matches:
            candidates = CATEGORY_WINDOW_CLASSES.get(clean_query, [])
            if not candidates:
                for cat, classes in CATEGORY_WINDOW_CLASSES.items():
                    if cat in clean_query:
                        candidates = classes
                        break

            for cand in candidates:
                cand_matches = self.kdotool.search(cand, limit=1)
                if cand_matches:
                    matches = cand_matches
                    break

        if not matches:
            return ToolResult(
                success=False,
                message=f"No open window matching '{query}' found.",
                data={"query": query},
                error="Window not found",
            )

        target = matches[0]
        self.kdotool.activate_window(target)
        win_name = self.kdotool.get_window_name(target)
        return ToolResult(
            success=True,
            message=f"Switched to window '{win_name}'.",
            data={"window_id": target, "name": win_name, "query": query},
            observation_required=True,
        )
