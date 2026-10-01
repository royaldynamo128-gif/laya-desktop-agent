"""
Application Catalog and Intelligent Entity Resolution Engine for Laya Desktop Agent.
Builds an indexed inventory of installed applications with full metadata,
phonetic hashing, descriptor stripping, semantic comment search, and safe ambiguity detection.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
import configparser
import difflib
import json
import logging
import os
import re
import shutil

logger = logging.getLogger(__name__)

DESKTOP_DIRS = [
    Path("/usr/share/applications"),
    Path("/var/lib/flatpak/exports/share/applications"),
    Path.home() / ".local/share/applications",
    Path.home() / ".local/share/flatpak/exports/share/applications",
]

# Generic descriptors that users append to application names in natural speech
GENERIC_DESCRIPTORS = [
    "browser",
    "web browser",
    "internet browser",
    "app",
    "application",
    "program",
    "tool",
    "client",
    "terminal",
    "terminal emulator",
    "editor",
    "text editor",
    "code editor",
    "player",
    "media player",
    "music player",
    "video player",
    "file manager",
    "files",
    "software",
    "utility",
]

# Strict 1:1 curated aliases (NEVER map generic words to specific proprietary apps!)
STRICT_ALIASES = {
    "google chrome": "com.google.Chrome.desktop",
    "chrome": "com.google.Chrome.desktop",
    "opera": "opera.desktop",
    "brave": "brave-browser.desktop",
    "brave browser": "brave-browser.desktop",
    "firefox": "org.mozilla.firefox.desktop",
    "mozilla firefox": "org.mozilla.firefox.desktop",
    "chromium": "org.chromium.Chromium.desktop",
    "ghostty": "com.mitchellh.ghostty.desktop",
    "konsole": "org.kde.konsole.desktop",
    "kitty": "kitty.desktop",
    "dolphin": "org.kde.dolphin.desktop",
    "telegram": "org.telegram.desktop.desktop",
    "telegram desktop": "org.telegram.desktop.desktop",
    "obsidian": "md.obsidian.Obsidian.desktop",
    "kcalc": "org.kde.kcalc.desktop",
    "calculator": "org.kde.kcalc.desktop",
    "vlc": "vlc.desktop",
    "upscayl": "org.upscayl.Upscayl.desktop",
    "settings": "systemsettings.desktop",
    "system settings": "systemsettings.desktop",
    "kde settings": "systemsettings.desktop",
    "plasma settings": "systemsettings.desktop",
    "notepad": "org.kde.kwrite.desktop",
    "note pad": "org.kde.kwrite.desktop",
    "not pad": "org.kde.kwrite.desktop",
    "notepadd": "org.kde.kwrite.desktop",
    "text editor": "org.kde.kwrite.desktop",
    "the text editor": "org.kde.kwrite.desktop",
    "a text editor": "org.kde.kwrite.desktop",
    "kate": "org.kde.kate.desktop",
    "kwrite": "org.kde.kwrite.desktop",
    "dolphin": "org.kde.dolphin.desktop",
    "dolfin": "org.kde.dolphin.desktop",
    "doll fin": "org.kde.dolphin.desktop",
    "dolphen": "org.kde.dolphin.desktop",
    "crome": "com.google.Chrome.desktop",
    "settings": "systemsettings.desktop",
    "set things": "systemsettings.desktop",
    "setings": "systemsettings.desktop",
}


def soundex(text: str) -> str:
    """Compute American Soundex phonetic code for English speech pronunciation."""
    clean = re.sub(r"[^A-Za-z]", "", text).upper()
    if not clean:
        return ""

    first_char = clean[0]
    mapping = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }

    encoded = []
    prev_code = mapping.get(first_char, "0")

    for char in clean[1:]:
        code = mapping.get(char, "0")
        if code != "0":
            if code != prev_code:
                encoded.append(code)
            prev_code = code
        else:
            prev_code = "0"

    code_str = first_char + "".join(encoded)
    return (code_str + "000")[:4]


@dataclass
class AppEntry:
    desktop_id: str
    name: str
    display_name: str
    exec_cmd: str
    executable: str
    path: str
    generic_name: str = ""
    comment: str = ""
    keywords: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    wm_class: str = ""
    candidate_wm_classes: List[str] = field(default_factory=list)
    phonetic_code: str = ""
    tokens: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "desktop_id": self.desktop_id,
            "name": self.name,
            "display_name": self.display_name,
            "exec_cmd": self.exec_cmd,
            "executable": self.executable,
            "path": self.path,
            "generic_name": self.generic_name,
            "comment": self.comment,
            "keywords": self.keywords,
            "categories": self.categories,
            "wm_class": self.wm_class,
            "candidate_wm_classes": self.candidate_wm_classes,
            "phonetic_code": self.phonetic_code,
        }


@dataclass
class ResolvedApp:
    name: str
    desktop_id: str
    exec_cmd: str
    wm_class: str
    confidence: float
    match_tier: str  # exact, descriptor_stripped, alias, phonetic, metadata, fuzzy, ambiguous, unresolved
    generic_name: str = ""
    comment: str = ""
    needs_clarification: bool = False
    clarification_prompt: str = ""
    candidates: List[str] = field(default_factory=list)
    candidate_wm_classes: List[str] = field(default_factory=list)


class AppCatalog:
    def __init__(self, cache_file: Optional[Path] = None):
        self.cache_file = cache_file
        self.apps: Dict[str, AppEntry] = {}  # desktop_id -> AppEntry
        self.name_to_id: Dict[str, str] = {}  # normalized name -> desktop_id
        self.exec_to_id: Dict[str, str] = {}  # binary name -> desktop_id
        self.phonetic_to_ids: Dict[str, List[str]] = {}  # soundex -> [desktop_ids]
        self._load_or_index()

    def _load_or_index(self) -> None:
        if self.cache_file and self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("apps", []):
                        item.setdefault("candidate_wm_classes", [item.get("wm_class", "")] if item.get("wm_class") else [])
                        entry = AppEntry(**item)
                        entry.tokens = set(entry.name.lower().split())
                        self.apps[entry.desktop_id] = entry
                        self.name_to_id[entry.name.lower()] = entry.desktop_id
                        if entry.executable:
                            self.exec_to_id[entry.executable.lower()] = entry.desktop_id
                        if entry.phonetic_code:
                            self.phonetic_to_ids.setdefault(entry.phonetic_code, []).append(entry.desktop_id)
                    if self.apps:
                        return
            except Exception as e:
                logger.debug("Failed to load app catalog cache: %s", e)

        self.reindex()

    def reindex(self) -> None:
        """Scan all desktop directories and build the full application catalog."""
        self.apps.clear()
        self.name_to_id.clear()
        self.exec_to_id.clear()
        self.phonetic_to_ids.clear()

        for d in DESKTOP_DIRS:
            if not d.exists():
                continue
            for entry_path in d.glob("*.desktop"):
                try:
                    cp = configparser.ConfigParser(interpolation=None)
                    cp.read(str(entry_path), encoding="utf-8")
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

                    desktop_id = entry_path.name
                    # Avoid overriding higher priority paths (e.g. user local over system)
                    if desktop_id in self.apps:
                        continue

                    generic_name = sec.get("GenericName", "").strip()
                    comment = sec.get("Comment", "").strip()
                    wm_class = sec.get("StartupWMClass", "").strip()
                    categories = [c.strip() for c in sec.get("Categories", "").split(";") if c.strip()]
                    keywords = [k.strip() for k in sec.get("Keywords", "").split(";") if k.strip()]

                    # Extract binary executable name
                    binary_match = re.match(r"^(\S+)", exec_cmd)
                    executable = os.path.basename(binary_match.group(1)) if binary_match else ""
                    if executable.startswith("flatpak") or executable == "env":
                        # If flatpak command, extract actual app binary if present
                        submatch = re.search(r"--command=(\S+)", exec_cmd)
                        if submatch:
                            executable = submatch.group(1)

                    desktop_stem = entry_path.stem.lower()
                    candidates: List[str] = []
                    if wm_class:
                        candidates.append(wm_class)
                    if desktop_stem not in [c.lower() for c in candidates]:
                        candidates.append(desktop_stem)
                    if executable and executable.lower() not in [c.lower() for c in candidates]:
                        candidates.append(executable.lower())

                    last_seg = desktop_stem.split(".")[-1]
                    if last_seg not in [c.lower() for c in candidates]:
                        candidates.append(last_seg)

                    KNOWN_WM_CLASSES = {
                        "chrome": ["google-chrome", "Google-chrome"],
                        "telegram": ["TelegramDesktop", "telegram-desktop"],
                        "systemsettings": ["systemsettings", "System Settings"],
                        "dolphin": ["org.kde.dolphin", "dolphin"],
                        "kwrite": ["kwrite", "org.kde.kwrite"],
                        "kate": ["kate", "org.kde.kate"],
                        "ghostty": ["com.mitchellh.ghostty", "ghostty"],
                        "konsole": ["org.kde.konsole", "konsole"],
                        "obsidian": ["obsidian"],
                    }
                    for kw, kclasses in KNOWN_WM_CLASSES.items():
                        if kw in desktop_stem or (executable and kw == executable.lower()):
                            for kc in kclasses:
                                if kc not in candidates:
                                    candidates.append(kc)

                    if not wm_class:
                        wm_class = candidates[0] if candidates else name

                    p_code = soundex(name)
                    tokens = set(name.lower().split())

                    app_entry = AppEntry(
                        desktop_id=desktop_id,
                        name=name,
                        display_name=name,
                        exec_cmd=exec_cmd,
                        executable=executable,
                        path=str(entry_path),
                        generic_name=generic_name,
                        comment=comment,
                        keywords=keywords,
                        categories=categories,
                        wm_class=wm_class,
                        candidate_wm_classes=candidates,
                        phonetic_code=p_code,
                        tokens=tokens,
                    )

                    self.apps[desktop_id] = app_entry
                    self.name_to_id[name.lower()] = desktop_id
                    if executable:
                        self.exec_to_id[executable.lower()] = desktop_id
                    if p_code:
                        self.phonetic_to_ids.setdefault(p_code, []).append(desktop_id)

                except Exception as e:
                    logger.debug("Error parsing %s: %s", entry_path, e)

        logger.info("Indexed %d desktop applications in AppCatalog.", len(self.apps))

        if self.cache_file:
            try:
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_file, "w", encoding="utf-8") as f:
                    json.dump({"apps": [a.to_dict() for a in self.apps.values()]}, f, indent=2)
            except Exception as e:
                logger.debug("Failed to write app catalog cache: %s", e)

    def resolve(self, spoken_target: str) -> Optional[ResolvedApp]:
        """
        Intelligently resolve spoken application name to a verified AppEntry.
        Strict Priority:
          1. Exact match (name / binary / stem)
          2. Descriptor stripping ("opera browser" -> "opera")
          3. Curated 1:1 alias
          4. Phonetic similarity (soundex: "upscale" -> "Upscayl")
          5. Semantic / metadata match ("image upscaler" -> Upscayl)
          6. Fuzzy edit distance (Levenshtein ratio >= 0.85)
          7. Ambiguity check -> needs_clarification
          8. None (DO NOT GUESS)
        """
        if not spoken_target:
            return None

        raw = spoken_target.strip()
        clean = raw.lower()

        # Remove leading conversational prefixes ("my ", "the ", "a ")
        clean = re.sub(r"^(my|the|a|an)\s+", "", clean).strip()

        # --- Tier 1: Exact Match ---
        # 1a. Name exact
        if clean in self.name_to_id:
            entry = self.apps[self.name_to_id[clean]]
            return self._build_resolved(entry, confidence=1.0, match_tier="exact")

        # 1b. Executable binary exact
        if clean in self.exec_to_id:
            entry = self.apps[self.exec_to_id[clean]]
            return self._build_resolved(entry, confidence=1.0, match_tier="exact_executable")

        # 1c. Desktop ID stem exact (e.g. "opera" -> opera.desktop)
        for app in self.apps.values():
            stem = Path(app.desktop_id).stem.lower()
            if clean == stem:
                return self._build_resolved(app, confidence=1.0, match_tier="exact_stem")
            if clean not in ["settings", "editor", "app", "ui", "view", "viewer", "client"]:
                if clean == stem.split(".")[-1]:
                    return self._build_resolved(app, confidence=0.95, match_tier="exact_stem")

        # --- Tier 2: Specific Descriptor Stripping ---
        # e.g. "opera browser" -> strip "browser" -> "opera"
        # "upscayl app" -> strip "app" -> "upscayl"
        # "ghostty terminal" -> strip "terminal" -> "ghostty"
        for desc in sorted(GENERIC_DESCRIPTORS, key=len, reverse=True):
            # Check trailing descriptor
            if clean.endswith(" " + desc):
                stripped = clean[:-len(" " + desc)].strip()
                if stripped:
                    res = self.resolve(stripped)
                    if res and res.confidence >= 0.85:
                        res.confidence = 0.96
                        res.match_tier = "descriptor_stripped"
                        return res
            # Check leading descriptor (e.g. "browser opera")
            if clean.startswith(desc + " "):
                stripped = clean[len(desc + " "):].strip()
                if stripped:
                    res = self.resolve(stripped)
                    if res and res.confidence >= 0.85:
                        res.confidence = 0.96
                        res.match_tier = "descriptor_stripped"
                        return res

        # --- Tier 3: Curated 1:1 Aliases & Core System Categories ---
        if clean in STRICT_ALIASES:
            target_id = STRICT_ALIASES[clean]
            if target_id in self.apps:
                return self._build_resolved(self.apps[target_id], confidence=0.95, match_tier="strict_alias")

        # System Settings
        if clean in ["settings", "system settings", "kde settings", "plasma settings", "control panel", "preferences"]:
            for sid in ["systemsettings.desktop", "org.kde.systemsettings.desktop", "gnome-control-center.desktop"]:
                if sid in self.apps:
                    return self._build_resolved(self.apps[sid], confidence=0.98, match_tier="system_settings")

        # Text Editor / Notepad
        if clean in ["notepad", "text editor", "a text editor", "editor", "code editor"]:
            for ed_id in ["org.kde.kwrite.desktop", "org.kde.kate.desktop", "org.gnome.TextEditor.desktop", "gedit.desktop"]:
                if ed_id in self.apps:
                    return self._build_resolved(self.apps[ed_id], confidence=0.95, match_tier="default_editor")

        # Handle standalone generic requests (e.g. "browser", "default browser", "terminal", "file manager")
        if clean in ["browser", "web browser", "default browser", "my browser", "internet"]:
            # Query default browser via xdg-settings
            default_browser_id = self._get_default_browser_desktop_id()
            if default_browser_id and default_browser_id in self.apps:
                return self._build_resolved(self.apps[default_browser_id], confidence=0.92, match_tier="system_default")
            for b_id in ["com.google.Chrome.desktop", "google-chrome.desktop", "opera.desktop", "firefox.desktop"]:
                if b_id in self.apps:
                    return self._build_resolved(self.apps[b_id], confidence=0.92, match_tier="default_browser")

        if clean in ["terminal", "my terminal", "console"]:
            # Prefer Ghostty if installed, then Konsole, then kitty
            for term_id in ["com.mitchellh.ghostty.desktop", "org.kde.konsole.desktop", "kitty.desktop", "alacritty.desktop"]:
                if term_id in self.apps:
                    return self._build_resolved(self.apps[term_id], confidence=0.92, match_tier="default_terminal")

        if clean in ["files", "file manager", "my files", "explorer", "file browser"]:
            for fm_id in ["org.kde.dolphin.desktop", "org.gnome.Nautilus.desktop", "thunar.desktop"]:
                if fm_id in self.apps:
                    return self._build_resolved(self.apps[fm_id], confidence=0.92, match_tier="default_files")

        # --- Tier 4: Phonetic / Soundex Similarity ---
        # e.g. "upscale" / "upscale app" -> Soundex U124 matches Upscayl (U124)
        # "oprah" -> Soundex O160 matches Opera (O160)
        target_soundex = soundex(clean)
        if target_soundex and target_soundex in self.phonetic_to_ids:
            matching_ids = self.phonetic_to_ids[target_soundex]
            if len(matching_ids) == 1:
                entry = self.apps[matching_ids[0]]
                # Verify that edit distance is within reason (e.g. ratio >= 0.65)
                ratio = difflib.SequenceMatcher(None, clean, entry.name.lower()).ratio()
                if ratio >= 0.60:
                    logger.info("Phonetic match: '%s' -> '%s' (soundex: %s, ratio: %.2f)", raw, entry.name, target_soundex, ratio)
                    return self._build_resolved(entry, confidence=0.88, match_tier="phonetic")
            elif len(matching_ids) > 1:
                # Disambiguate by edit distance
                scored = []
                for mid in matching_ids:
                    app = self.apps[mid]
                    r = difflib.SequenceMatcher(None, clean, app.name.lower()).ratio()
                    scored.append((r, app))
                scored.sort(key=lambda x: x[0], reverse=True)
                if scored[0][0] >= 0.70 and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.15):
                    return self._build_resolved(scored[0][1], confidence=0.86, match_tier="phonetic")

        # Check individual words in multi-word queries (e.g. "image upscaler" -> "upscaler" Soundex U124)
        words = clean.split()
        if len(words) > 1:
            for w in words:
                w_soundex = soundex(w)
                if w_soundex and w_soundex in self.phonetic_to_ids:
                    for mid in self.phonetic_to_ids[w_soundex]:
                        app = self.apps[mid]
                        other_words = [ow for ow in words if ow != w]
                        text_corpus = f"{app.name} {app.generic_name} {app.comment} {' '.join(app.keywords)} {' '.join(app.categories)}".lower()
                        if all(ow in text_corpus or difflib.SequenceMatcher(None, ow, app.name.lower()).ratio() >= 0.7 for ow in other_words):
                            logger.info("Multi-word phonetic match: '%s' (word '%s' soundex: %s) -> '%s'", raw, w, w_soundex, app.name)
                            return self._build_resolved(app, confidence=0.88, match_tier="multiword_phonetic")

        # --- Tier 5: Semantic Metadata & Comment Match ---
        # e.g. "image upscaler" -> matches comment="Upscale Images with AI" on Upscayl
        def _token_stems(word: str) -> List[str]:
            w = word.lower()
            stems = [w]
            s = re.sub(r"(er|ing|ed|s)$", "", w)
            if s and s != w:
                stems.append(s)
            return stems

        metadata_matches = []
        for app in self.apps.values():
            text_corpus = f"{app.name} {app.generic_name} {app.comment} {' '.join(app.keywords)} {' '.join(app.categories)}".lower()
            if clean in text_corpus:
                metadata_matches.append(app)
            else:
                all_tokens_match = True
                for tok in clean.split():
                    stems = _token_stems(tok)
                    tok_soundex = soundex(tok)
                    tok_match = any(st in text_corpus for st in stems) or (tok_soundex == app.phonetic_code)
                    if not tok_match:
                        all_tokens_match = False
                        break
                if all_tokens_match:
                    metadata_matches.append(app)

        if len(metadata_matches) == 1:
            logger.info("Semantic metadata match: '%s' -> '%s'", raw, metadata_matches[0].name)
            return self._build_resolved(metadata_matches[0], confidence=0.84, match_tier="metadata_semantic")
        elif len(metadata_matches) > 1:
            # Check if one of them is an exact category match
            exact_cat = [a for a in metadata_matches if clean == a.generic_name.lower()]
            if len(exact_cat) == 1:
                return self._build_resolved(exact_cat[0], confidence=0.85, match_tier="generic_name_exact")
            elif len(exact_cat) > 1:
                # Prefer GUI apps over console/terminal tools
                gui_apps = [a for a in exact_cat if a.executable not in ("vim", "vi", "nano") and "ConsoleOnly" not in a.categories]
                if gui_apps:
                    kde_apps = [a for a in gui_apps if "kde" in a.desktop_id.lower() or "kde" in [c.lower() for c in a.categories]]
                    chosen = kde_apps[0] if kde_apps else gui_apps[0]
                    return self._build_resolved(chosen, confidence=0.85, match_tier="generic_name_gui")
                return self._build_resolved(exact_cat[0], confidence=0.82, match_tier="generic_name_fallback")

        # --- Tier 6: Fuzzy String Matching (Levenshtein Ratio) ---
        candidates = []
        for app in self.apps.values():
            ratio = difflib.SequenceMatcher(None, clean, app.name.lower()).ratio()
            if ratio >= 0.78:
                candidates.append((ratio, app))

        candidates.sort(key=lambda x: x[0], reverse=True)
        if candidates:
            top_ratio, top_app = candidates[0]
            if top_ratio >= 0.82:
                # Check if there is a runner-up with nearly identical score (ambiguity)
                if len(candidates) > 1 and (top_ratio - candidates[1][0]) < 0.08:
                    candidate_names = [c[1].name for c in candidates[:3]]
                    logger.warning("Ambiguous match for '%s': candidates=%s", raw, candidate_names)
                    return ResolvedApp(
                        name=top_app.name,
                        desktop_id=top_app.desktop_id,
                        exec_cmd=top_app.exec_cmd,
                        wm_class=top_app.wm_class,
                        confidence=0.65,
                        match_tier="ambiguous",
                        needs_clarification=True,
                        clarification_prompt=f"Did you mean {candidate_names[0]} or {candidate_names[1]}?",
                        candidates=candidate_names,
                        candidate_wm_classes=list(top_app.candidate_wm_classes) if hasattr(top_app, "candidate_wm_classes") else [top_app.wm_class],
                    )
                return self._build_resolved(top_app, confidence=top_ratio, match_tier="fuzzy")

        # --- Tier 7: Direct Executable in PATH (Requirement 7) ---
        which_bin = shutil.which(clean)
        if which_bin and os.access(which_bin, os.X_OK):
            bin_name = os.path.basename(which_bin)
            logger.info("Direct executable match from PATH for '%s': %s", raw, which_bin)
            return ResolvedApp(
                name=bin_name,
                desktop_id="",
                exec_cmd=which_bin,
                wm_class=bin_name,
                confidence=0.80,
                match_tier="executable_path",
                generic_name="Executable",
                comment=f"Direct binary from PATH: {which_bin}",
                candidate_wm_classes=[bin_name],
            )

        # --- Tier 8: Rejection (NEVER GUESS) ---
        logger.info("Application resolution failed for '%s' (no confident match found).", raw)
        return None

    def _build_resolved(self, entry: AppEntry, confidence: float, match_tier: str) -> ResolvedApp:
        candidates = list(entry.candidate_wm_classes) if getattr(entry, "candidate_wm_classes", None) else ([entry.wm_class] if entry.wm_class else [entry.name])
        return ResolvedApp(
            name=entry.name,
            desktop_id=entry.desktop_id,
            exec_cmd=entry.exec_cmd,
            wm_class=entry.wm_class,
            confidence=round(confidence, 3),
            match_tier=match_tier,
            generic_name=entry.generic_name,
            comment=entry.comment,
            needs_clarification=False,
            clarification_prompt="",
            candidates=[],
            candidate_wm_classes=candidates,
        )

    def _get_default_browser_desktop_id(self) -> Optional[str]:
        """Query system default browser via xdg-settings."""
        try:
            res = os.popen("xdg-settings get default-web-browser 2>/dev/null").read().strip()
            if res:
                return res
            res_mime = os.popen("xdg-mime query default x-scheme-handler/https 2>/dev/null").read().strip()
            if res_mime:
                return res_mime
        except Exception:
            pass
        return None

    def generate_whisper_prompt(self) -> str:
        """Build dynamic initial prompt biasing Faster-Whisper decoder with natural command sentences."""
        return (
            "Open Chrome. Open Dolphin. Open the text editor and write Hello World. "
            "Open Chrome and search for Linux. Open Settings. Switch to Telegram. "
            "Open Ghostty. Close this window. Take a screenshot."
        )

    def generate_hotwords(self) -> str:
        """Generate space-separated hotwords string to boost token probabilities in Faster-Whisper."""
        words = {"Open", "Close", "Launch", "Switch", "Minimize", "Maximize", "Terminal", "Chrome", "Opera", "Upscayl", "Ghostty", "Telegram", "Dolphin", "Brave"}
        for app in self.apps.values():
            first_token = app.name.split()[0].strip()
            if first_token.isalnum() and len(first_token) > 2:
                words.add(first_token)
        return " ".join(sorted(words))
