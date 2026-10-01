"""
Application Control Tools for Laya Desktop Agent.
Handles opening, closing, switching, and focusing desktop applications and directory folders.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
from tools.base import Tool, ToolResult, SafetyTier
from integrations.flatpak_helper import AppLauncher
from integrations.kdotool_client import KdotoolClient
from integrations.app_catalog import AppCatalog

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

    def __init__(self, launcher: AppLauncher, kdotool: KdotoolClient, catalog: Optional[AppCatalog] = None):
        self.launcher = launcher
        self.kdotool = kdotool
        self.catalog = catalog or AppCatalog()

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

    def _sanitize_exec_cmd(self, exec_cmd: str) -> List[str]:
        # Strip field codes like %f, %F, %u, %U and flatpak @@u %U @@
        s = re.sub(r"@@u\s+%[a-zA-Z]\s+@@", "", exec_cmd)
        s = re.sub(r"@@[a-zA-Z]\s+%[a-zA-Z]\s+@@", "", s)
        s = re.sub(r"%[a-zA-Z]", "", s)
        tokens = shlex.split(s.strip())
        while tokens and tokens[-1] == "--":
            tokens.pop()
        return tokens

    def execute(
        self,
        name: Optional[str] = None,
        app_name: Optional[str] = None,
        desktop_id: Optional[str] = None,
        wm_class: Optional[str] = None,
        exec_cmd: Optional[str] = None,
        candidate_wm_classes: Optional[List[str]] = None,
        **kwargs
    ) -> ToolResult:
        target_name = (name or app_name or "").strip()
        if not target_name:
            return ToolResult(success=False, message="Application or folder name required.", error="Missing name parameter")

        # 1. Check if target is a known directory/folder
        target_folder = self._detect_folder(target_name)
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

        # 2. Application Resolution
        target_desktop_id = desktop_id
        target_exec_cmd = exec_cmd
        target_wm_class = wm_class
        candidate_classes = list(candidate_wm_classes) if candidate_wm_classes else []
        canonical_name = target_name
        desktop_file_path = None
        executable = None

        resolved = self.catalog.resolve(target_name)
        if resolved:
            canonical_name = resolved.name
            target_desktop_id = target_desktop_id or resolved.desktop_id
            target_wm_class = target_wm_class or resolved.wm_class
            target_exec_cmd = target_exec_cmd or resolved.exec_cmd
            for c in getattr(resolved, "candidate_wm_classes", []):
                if c not in candidate_classes:
                    candidate_classes.append(c)
            if target_desktop_id and target_desktop_id in self.catalog.apps:
                desktop_file_path = self.catalog.apps[target_desktop_id].path
                executable = self.catalog.apps[target_desktop_id].executable

        if target_wm_class and target_wm_class not in candidate_classes:
            candidate_classes.insert(0, target_wm_class)
        if canonical_name not in candidate_classes:
            candidate_classes.append(canonical_name)
        if target_name not in candidate_classes:
            candidate_classes.append(target_name)

        # 3. Check if application is already running, switch and bring to front if so
        running_win = None
        for cand in candidate_classes:
            if not cand or not cand.strip():
                continue
            matches = self.kdotool.search(cand.strip(), limit=5)
            for m in matches:
                cls_name = self.kdotool.get_window_class(m).lower()
                if cls_name and cls_name != "plasmashell":
                    running_win = m
                    break
            if running_win:
                break

        if running_win:
            logger.info("Found already running window %s for %s. Activating...", running_win, canonical_name)
            self.kdotool.activate_window(running_win)
            time.sleep(0.2)
            title = self.kdotool.get_window_name(running_win)
            return ToolResult(
                success=True,
                message=f"Switched to running {canonical_name}.",
                data={
                    "window_id": running_win,
                    "title": title,
                    "already_running": True,
                    "wm_class": target_wm_class or "",
                    "name": canonical_name,
                },
                observation_required=True,
            )

        # 4. Multi-tier application launching
        pre_existing_windows = set(self.kdotool.search("", limit=50))
        launched = False
        launch_error = None
        launch_mechanism = ""

        # Tier 1: KDE / Desktop Standard Launchers
        # 1a. kstart (native KDE Plasma 6)
        if not launched and target_desktop_id and shutil.which("kstart"):
            try:
                app_stem = target_desktop_id[:-8] if target_desktop_id.endswith(".desktop") else target_desktop_id
                cmd = ["kstart", "--application", app_stem]
                logger.info("Launching %s via kstart: %s", canonical_name, cmd)
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                launched = True
                launch_mechanism = "kstart"
            except Exception as e:
                logger.debug("kstart failed for %s: %s", canonical_name, e)
                launch_error = str(e)

        # 1b. gtk-launch
        if not launched and target_desktop_id and shutil.which("gtk-launch"):
            try:
                cmd = ["gtk-launch", target_desktop_id]
                logger.info("Launching %s via gtk-launch: %s", canonical_name, cmd)
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                launched = True
                launch_mechanism = "gtk-launch"
            except Exception as e:
                logger.debug("gtk-launch failed for %s: %s", canonical_name, e)
                launch_error = str(e)

        # 1c. gio launch
        if not launched and desktop_file_path and shutil.which("gio"):
            try:
                cmd = ["gio", "launch", desktop_file_path]
                logger.info("Launching %s via gio: %s", canonical_name, cmd)
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                launched = True
                launch_mechanism = "gio"
            except Exception as e:
                logger.debug("gio launch failed for %s: %s", canonical_name, e)
                launch_error = str(e)

        # Tier 2: Direct execution of sanitized exec_cmd
        if not launched and target_exec_cmd:
            try:
                cmd = self._sanitize_exec_cmd(target_exec_cmd)
                if cmd:
                    logger.info("Launching %s via sanitized exec: %s", canonical_name, cmd)
                    subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    launched = True
                    launch_mechanism = "sanitized_exec"
            except Exception as e:
                logger.debug("Direct exec failed for %s (%s): %s", canonical_name, target_exec_cmd, e)
                launch_error = str(e)

        # Tier 3: Direct PATH binary execution
        if not launched:
            candidate_bin = executable or target_name.lower().split()[0]
            bin_path = shutil.which(candidate_bin)
            if bin_path and os.access(bin_path, os.X_OK):
                try:
                    logger.info("Launching %s via PATH binary: %s", canonical_name, bin_path)
                    subprocess.Popen([bin_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                    launched = True
                    launch_mechanism = "binary_path"
                except Exception as e:
                    logger.debug("PATH binary launch failed for %s: %s", canonical_name, e)
                    launch_error = str(e)

        # Tier 4: Fallback to AppLauncher / xdg-open
        if not launched:
            success, msg = self.launcher.launch(target_name)
            if success:
                launched = True
                launch_mechanism = "app_launcher"
            else:
                launch_error = msg

        if not launched:
            return ToolResult(
                success=False,
                message=f"Failed to launch {canonical_name}: {launch_error}",
                error=launch_error,
                observation_required=False,
            )

        # 5. Bring newly launched window to focus and verify window detection (Requirement 8)
        new_window_id = None
        for _ in range(28):
            time.sleep(0.25)
            # 1. Check for newly spawned windows
            current_windows = self.kdotool.search("", limit=50)
            diff_windows = [w for w in current_windows if w not in pre_existing_windows]
            for w in diff_windows:
                cls_name = self.kdotool.get_window_class(w).lower()
                if cls_name and cls_name != "plasmashell":
                    new_window_id = w
                    break
            if new_window_id:
                break

            # 2. Search by candidate classes and names
            for cand in candidate_classes:
                if not cand or not cand.strip():
                    continue
                matches = self.kdotool.search(cand.strip(), limit=5)
                for m in matches:
                    cls_name = self.kdotool.get_window_class(m).lower()
                    if cls_name and cls_name != "plasmashell":
                        new_window_id = m
                        break
                if new_window_id:
                    break
            if new_window_id:
                break

        if not new_window_id:
            logger.warning("Window for %s was not detected within timeout.", canonical_name)
            return ToolResult(
                success=False,
                message=f"Application '{canonical_name}' launched via {launch_mechanism}, but its window was not detected.",
                error="WindowNotDetected",
                data={"target_name": canonical_name, "launch_mechanism": launch_mechanism},
                observation_required=False,
            )

        # Activate the detected window
        self.kdotool.activate_window(new_window_id)
        time.sleep(0.15)
        title = self.kdotool.get_window_name(new_window_id)

        return ToolResult(
            success=True,
            message=f"Launched {canonical_name}.",
            data={
                "target_name": canonical_name,
                "window_id": new_window_id,
                "title": title,
                "desktop_id": target_desktop_id or "",
                "wm_class": target_wm_class or "",
                "launch_mechanism": launch_mechanism,
                "already_running": False,
            },
            observation_required=True,
        )


class CloseApplicationTool(Tool):
    name = "close_application"
    description = "Close or quit a window or application"
    safety_tier = SafetyTier.SAFE

    def __init__(self, kdotool: KdotoolClient):
        self.kdotool = kdotool

    def execute(
        self,
        name: Optional[str] = None,
        wm_class: Optional[str] = None,
        window_id: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        target_win = window_id
        if not target_win:
            if wm_class:
                matches = self.kdotool.search(wm_class, limit=1)
                if matches:
                    target_win = matches[0]
            if not target_win and name:
                matches = self.kdotool.search(name, limit=1)
                if matches:
                    target_win = matches[0]
            elif not target_win:
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

    def execute(
        self,
        query: str = "",
        wm_class: Optional[str] = None,
        candidate_wm_classes: Optional[list] = None,
        **kwargs
    ) -> ToolResult:
        if not query and not wm_class and not candidate_wm_classes:
            # Switch to previous window
            all_wins = self.kdotool.search("", limit=10)
            valid_wins = [w for w in all_wins if self.kdotool.get_window_class(w).lower() != "plasmashell"]
            if len(valid_wins) >= 2:
                target = valid_wins[1]
                self.kdotool.activate_window(target)
                name = self.kdotool.get_window_name(target)
                return ToolResult(
                    success=True,
                    message=f"Switched to previous window '{name}'.",
                    data={"window_id": target, "name": name},
                    observation_required=True,
                )
            return ToolResult(success=False, message="No other windows available.", error="Stack empty")

        clean_query = re.sub(r"^(the|my|a|an)\s+", "", query.lower().strip()) if query else ""

        # 1. Search by wm_class or candidate_wm_classes if provided
        matches = []
        cands = list(candidate_wm_classes or kwargs.get("candidate_wm_classes") or [])
        if wm_class and wm_class not in cands:
            cands.insert(0, wm_class)

        for cand in cands:
            if not cand or not cand.strip():
                continue
            cand_matches = self.kdotool.search(cand.strip(), limit=5)
            filtered = [m for m in cand_matches if self.kdotool.get_window_class(m).lower() != "plasmashell"]
            if filtered:
                matches = filtered
                break

        # 2. Search by query string
        if not matches and clean_query:
            cand_matches = self.kdotool.search(clean_query, limit=5)
            filtered = [m for m in cand_matches if self.kdotool.get_window_class(m).lower() != "plasmashell"]
            if filtered:
                matches = filtered

        # 3. Check category mapping if direct search yields no matches
        if not matches and clean_query:
            category_candidates = CATEGORY_WINDOW_CLASSES.get(clean_query, [])
            if not category_candidates:
                for cat, classes in CATEGORY_WINDOW_CLASSES.items():
                    if cat in clean_query:
                        category_candidates = classes
                        break

            for cand in category_candidates:
                cand_matches = self.kdotool.search(cand, limit=5)
                filtered = [m for m in cand_matches if self.kdotool.get_window_class(m).lower() != "plasmashell"]
                if filtered:
                    matches = filtered
                    break

        if not matches:
            return ToolResult(
                success=False,
                message=f"No open window matching '{query or wm_class}' found.",
                data={"query": query, "wm_class": wm_class or ""},
                error="Window not found",
            )

        target = matches[0]
        self.kdotool.activate_window(target)
        win_name = self.kdotool.get_window_name(target)
        return ToolResult(
            success=True,
            message=f"Switched to window '{win_name}'.",
            data={"window_id": target, "name": win_name, "query": query, "wm_class": wm_class or ""},
            observation_required=True,
        )
