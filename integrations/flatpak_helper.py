"""
Application Indexer and Launcher for Native and Flatpak Desktop Applications.
Discovers desktop entries, provides fuzzy matching, and launches via gtk-launch / gio launch.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import configparser
import json
import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)

DESKTOP_DIRS = [
    Path("/usr/share/applications"),
    Path("/var/lib/flatpak/exports/share/applications"),
    Path.home() / ".local/share/applications",
    Path.home() / ".local/share/flatpak/exports/share/applications",
]

# Aliases and common voice terms for desktop apps
STANDARD_ALIASES = {
    "chrome": ["google chrome", "chrome", "com.google.chrome"],
    "google chrome": ["google chrome", "chrome"],
    "opera": ["opera", "opera browser"],
    "brave": ["brave", "brave browser"],
    "firefox": ["firefox", "mozilla firefox"],
    "telegram": ["telegram", "telegram desktop", "org.telegram.desktop"],
    "terminal": ["ghostty", "konsole", "terminal"],
    "ghostty": ["ghostty", "terminal"],
    "konsole": ["konsole", "terminal"],
    "files": ["dolphin", "files", "file manager", "org.kde.dolphin"],
    "dolphin": ["dolphin", "files"],
    "obsidian": ["obsidian", "notes", "md.obsidian.obsidian"],
    "vlc": ["vlc", "vlc media player", "media player"],
    "calculator": ["kcalc", "calculator"],
    "settings": ["system settings", "settings", "systemsettings"],
    "system settings": ["system settings", "settings", "systemsettings"],
    "kde settings": ["system settings", "settings", "systemsettings"],
    "notepad": ["kwrite", "kate", "notepad", "text editor"],
    "text editor": ["kwrite", "kate", "text editor"],
}


class AppLauncher:
    def __init__(self, cache_file: Optional[Path] = None):
        self.cache_file = cache_file
        self.app_index: Dict[str, Dict[str, str]] = {}
        self.name_map: Dict[str, str] = {}  # lowercase keyword -> desktop file basename
        self._load_or_index()

    def _load_or_index(self) -> None:
        if self.cache_file and self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.app_index = data.get("apps", {})
                    self.name_map = data.get("name_map", {})
                    if self.app_index:
                        return
            except Exception as e:
                logger.debug("Failed to load cached app registry: %s", e)

        self.reindex()

    def reindex(self) -> None:
        """Scan all desktop directories and build the application index."""
        logger.info("Indexing installed desktop applications...")
        self.app_index.clear()
        self.name_map.clear()

        for d in DESKTOP_DIRS:
            if not d.exists():
                continue
            for entry in d.glob("*.desktop"):
                try:
                    cp = configparser.ConfigParser(interpolation=None)
                    cp.read(str(entry), encoding="utf-8")
                    if not cp.has_section("Desktop Entry"):
                        continue
                    sec = cp["Desktop Entry"]
                    if sec.get("NoDisplay", "false").lower() == "true":
                        continue
                    if sec.get("Type", "Application") != "Application":
                        continue

                    name = sec.get("Name", "").strip()
                    exec_cmd = sec.get("Exec", "").strip()
                    if not name or not exec_cmd:
                        continue

                    basename = entry.name
                    self.app_index[basename] = {
                        "name": name,
                        "exec": exec_cmd,
                        "path": str(entry),
                        "basename": basename,
                    }

                    # Map primary name
                    clean_name = name.lower()
                    self.name_map[clean_name] = basename
                    # Map filename stem
                    stem = entry.stem.lower()
                    self.name_map[stem] = basename

                except Exception as e:
                    logger.debug("Could not parse %s: %s", entry, e)

        # Apply aliases
        for alias, targets in STANDARD_ALIASES.items():
            for t in targets:
                for k, v in self.app_index.items():
                    if t in v["name"].lower() or t in k.lower():
                        self.name_map[alias] = k
                        break
                if alias in self.name_map:
                    break

        logger.info("Indexed %d desktop applications.", len(self.app_index))
        if self.cache_file:
            try:
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_file, "w", encoding="utf-8") as f:
                    json.dump({"apps": self.app_index, "name_map": self.name_map}, f, indent=2)
            except Exception as e:
                logger.debug("Could not write app cache: %s", e)

    def find_app(self, query: str) -> Optional[Dict[str, str]]:
        """Find an application by query string (exact or fuzzy)."""
        q = query.lower().strip()
        
        # 1. Direct name_map hit
        if q in self.name_map:
            basename = self.name_map[q]
            return self.app_index.get(basename)

        # 2. Check aliases
        for alias, targets in STANDARD_ALIASES.items():
            if alias in q or any(t in q for t in targets):
                if alias in self.name_map:
                    return self.app_index.get(self.name_map[alias])

        # 3. Substring match on app names
        for basename, info in self.app_index.items():
            if q in info["name"].lower() or q in basename.lower():
                return info

        return None

    def launch(self, query: str) -> Tuple[bool, str]:
        """Launch application using gtk-launch or gio launch."""
        app = self.find_app(query)
        if not app:
            logger.warning("No application found matching query: '%s'", query)
            return False, f"Application '{query}' not found."

        basename = app["basename"]
        path = app["path"]

        # Prefer gtk-launch, fallback to gio launch
        launcher = shutil.which("gtk-launch")
        if launcher:
            cmd = [launcher, basename]
        else:
            launcher = shutil.which("gio")
            if launcher:
                cmd = [launcher, "launch", path]
            else:
                cmd = ["xdg-open", path]

        try:
            logger.info("Launching %s via %s", app["name"], cmd)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True, f"Launched {app['name']}."
        except Exception as e:
            logger.error("Failed to launch %s: %s", app["name"], e)
            return False, f"Failed to launch {app['name']}: {e}"
