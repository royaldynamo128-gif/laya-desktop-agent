"""
Generalized Semantic Parser for Laya Desktop Agent.

This is the natural-language understanding core. It is NOT a command
dictionary: it is a verb-frame parser with linguistic coordination
analysis that works for arbitrary applications, websites, dialogs,
buttons, files and future capabilities, because all entity resolution
(app names, targets, queries) is delegated to dynamic system state.

Key generalization mechanisms:
  1. Verb-anchored clause segmentation — text is split at coordination
     boundaries ONLY when what follows is itself a verb phrase. This makes
     "search for cats and dogs" one query while "open chrome and search
     cats" is two clauses.
  2. Verb inheritance / list distribution — "open chrome and opera"
     distributes the verb over coordinated noun phrases.
  3. Slot-based verb frames — each intent is a frame (verb + extractor),
     never a hardcoded command string.
  4. Referring expression detection — "this file", "it", "the image" are
     marked for context resolution by the executor.
  5. Quoted-string protection for type/search payloads.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verb lexicon (general verbs, NOT app-specific commands)
# ---------------------------------------------------------------------------

OPEN_VERBS = ["open", "launch", "start", "fire up", "run", "load", "boot", "bring up", "show me"]
CLOSE_VERBS = ["close", "quit", "exit", "shut down", "shut", "terminate", "kill"]
SWITCH_VERBS = ["switch to", "focus", "focus on", "bring to front", "activate", "go to", "bring up"]
SEARCH_VERBS = ["search for", "search", "look up", "lookup", "google", "find me", "find", "youtube", "check online for"]
NAVIGATE_VERBS = ["go to", "visit", "navigate to", "browse to", "open up", "load"]
TYPE_VERBS = ["type", "write", "enter", "input", "dictate", "say"]
PRESS_VERBS = ["press", "hit", "tap", "push", "key"]
CLICK_VERBS = ["click", "click on", "click the", "press the", "select", "choose", "pick", "check", "tick", "toggle", "enable", "hit the"]
SCROLL_VERBS = ["scroll", "page down", "page up", "scroll down", "scroll up"]
WAIT_VERBS = ["wait for", "wait until", "wait", "hold on"]
DOWNLOAD_VERBS = ["download", "grab", "fetch", "save from web"]
SAVE_VERBS = ["save as", "save it as", "save", "export as", "export", "convert to", "convert"]
SUBMIT_VERBS = ["submit", "send", "confirm", "apply", "proceed", "continue", "next", "finish", "done", "go ahead", "ok"]
FILL_VERBS = ["fill", "fill in", "fill out", "fill the", "complete"]

# Interaction verbs: unknown-app-specific UI operations resolved via perception
# (AT-SPI element match, vision/OCR). These make arbitrary app workflows possible
# without hardcoding any specific app's commands.
INTERACTION_VERBS = [
    "add", "import", "insert", "upload", "attach", "resize", "crop", "rotate",
    "flip", "scale", "apply", "adjust", "set", "increase", "decrease", "zoom",
    "install", "play this", "watch", "listen to", "follow", "subscribe",
]

MOVE_VERBS = ["move", "transfer", "put", "send to"]
COPY_VERBS = ["copy", "duplicate"]
RENAME_VERBS = ["rename", "call it", "name it"]
DELETE_VERBS = ["delete", "remove", "trash", "get rid of", "erase", "shred"]
CREATE_VERBS = ["create", "make", "new", "create a", "make a"]
COMPRESS_VERBS = ["compress", "zip", "archive", "pack"]
EXTRACT_VERBS = ["extract", "unzip", "unpack", "decompress"]
OPENFOLDER_VERBS = ["open", "show", "browse"]
VERIFY_VERBS = ["verify", "check", "make sure", "confirm that", "ensure"]

# Nouns that refer to files rather than applications (for open-intent disambiguation)
FILE_TYPE_NOUNS = {
    "image", "picture", "photo", "file", "document", "doc", "pdf",
    "video", "audio", "song", "music", "download", "archive", "zip",
    "screenshot", "spreadsheet", "presentation",
}

# Nouns that refer to content inside another application's UI
CONTENT_TYPE_NOUNS = {
    "game", "channel", "video", "playlist", "note", "chat", "conversation",
    "message", "project", "board", "repository", "repo", "tab", "page",
}

# All verbs that can begin a new coordinated clause (for segmentation)
ALL_VERB_PREFIXES: List[str] = sorted(
    set(
        OPEN_VERBS + CLOSE_VERBS + SWITCH_VERBS + SEARCH_VERBS + NAVIGATE_VERBS
        + TYPE_VERBS + PRESS_VERBS + CLICK_VERBS + SCROLL_VERBS + WAIT_VERBS
        + DOWNLOAD_VERBS + SAVE_VERBS + SUBMIT_VERBS + FILL_VERBS
        + MOVE_VERBS + COPY_VERBS + RENAME_VERBS + DELETE_VERBS + CREATE_VERBS
        + COMPRESS_VERBS + EXTRACT_VERBS
        + [
            "minimize", "maximise", "maximize", "fullscreen", "full screen", "restore", "tile",
            "screenshot", "take a screenshot", "take screenshot", "capture",
            "read", "what is on", "what's on", "describe",
            "mute", "unmute", "set volume", "volume", "turn up", "turn down", "play", "pause",
            "skip", "stop", "lock", "sleep", "shutdown", "reboot", "restart",
            "fill", "fill in", "fill out", "complete",
        ]
        + INTERACTION_VERBS
        + VERIFY_VERBS,
    ),
    key=len,
    reverse=True,
)

FILLER_WORDS = [
    "um", "uh", "er", "ah", "like, you know", "you know", "please", "could you",
    "can you", "would you", "will you", "for me", "hey laya", "laya", "hi laya",
]

LEADING_POLITENESS = [
    r"^(?:hey|hi|ok|okay|yo)\s+(?:laya|computer|assistant)?[,!]?\s*",
    r"^(?:can|could|would|will)\s+you\s+(?:please\s+)?",
    r"^i\s+(?:want|need|would like)\s+to\s+",
    r"^(?:please\s+)?",
]

CONJUNCTION_PATTERN = re.compile(
    r"\s*(?:,|;|\band then\b|\bthen\b|\bafter that\b|\band also\b|\bplus\b|\band\b)\s*",
    re.IGNORECASE,
)

URL_PATTERN = re.compile(
    r"^(?:https?://\S+|(?:www\.)?\S+\.(?:com|org|net|io|dev|ai|tv|me|gg|co|edu|gov|xyz|app|sh|us|uk|de|fr|jp|ru|br|in)\S*)$",
    re.IGNORECASE,
)

KNOWN_SITE_NAMES = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "google maps": "https://maps.google.com",
    "maps": "https://maps.google.com",
    "whatsapp": "https://web.whatsapp.com",
    "whatsapp web": "https://web.whatsapp.com",
    "telegram web": "https://web.telegram.org",
    "wikipedia": "https://www.wikipedia.org",
    "reddit": "https://www.reddit.com",
    "twitter": "https://twitter.com",
    "x": "https://x.com",
    "github": "https://github.com",
    "stack overflow": "https://stackoverflow.com",
    "stackoverflow": "https://stackoverflow.com",
    "netflix": "https://www.netflix.com",
    "spotify": "https://open.spotify.com",
    "twitch": "https://www.twitch.tv",
    "facebook": "https://www.facebook.com",
    "instagram": "https://www.instagram.com",
    "linkedin": "https://www.linkedin.com",
    "amazon": "https://www.amazon.com",
    "discord": "https://discord.com/app",
    "figma": "https://www.figma.com",
    "notion": "https://www.notion.so",
    "chatgpt": "https://chat.openai.com",
    "duckduckgo": "https://duckduckgo.com",
}

KEY_NAME_MAP = {
    "enter": "Return", "return": "Return", "esc": "Escape", "escape": "Escape",
    "tab": "Tab", "space": "space", "backspace": "BackSpace",
    "up": "Up", "down": "Down", "left": "Left", "right": "Right",
    "page down": "Page_Down", "page up": "Page_Up", "home": "Home", "end": "End",
    "delete": "Delete", "del": "Delete", "insert": "Insert",
    "ctrl": "ctrl", "control": "ctrl", "alt": "alt", "shift": "shift", "super": "super", "meta": "super", "win": "super",
    "f1": "F1", "f2": "F2", "f3": "F3", "f4": "F4", "f5": "F5", "f6": "F6",
    "f7": "F7", "f8": "F8", "f9": "F9", "f10": "F10", "f11": "F11", "f12": "F12",
}

WINDOW_ACTION_MAP = {
    "minimize": "minimize", "minimise": "minimize", "hide": "minimize",
    "maximize": "maximize", "maximise": "maximize", "full screen": "fullscreen", "fullscreen": "fullscreen",
    "restore": "restore", "tile": "tile", "snap": "tile",
}

REFERRING_EXPRESSIONS = {
    "this", "that", "it", "this file", "that file", "this image", "that image",
    "this folder", "this app", "this application", "this window", "this game",
    "this page", "this website", "the file", "the image", "the folder",
    "the download", "my download", "the latest download", "this video",
}

DIRECTION_WORDS = {"up", "down", "left", "right", "top", "bottom"}

FOLDER_WORDS = {
    "downloads": "Downloads", "download": "Downloads", "documents": "Documents",
    "document": "Documents", "desktop": "Desktop", "pictures": "Pictures",
    "picture": "Pictures", "photos": "Pictures", "music": "Music", "videos": "Videos",
    "video": "Videos", "projects": "Projects", "project": "Projects",
    "wallpapers": "Wallpapers", "wallpaper": "Wallpapers", "home": "~",
}


@dataclass
class Clause:
    """One interpreted clause of the user's utterance."""
    raw: str
    intent: str = "unknown"
    verb: str = ""
    target: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    is_referring: bool = False          # target is a referring expression ("this image")
    quoted_payload: Optional[str] = None  # exact text within quotes for type/search
    confidence: float = 0.9
    is_terminal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw": self.raw, "intent": self.intent, "verb": self.verb,
            "target": self.target, "arguments": self.arguments,
            "is_referring": self.is_referring,
        }


@dataclass
class ParseResult:
    raw: str
    normalized: str
    clauses: List[Clause]

    @property
    def is_compound(self) -> bool:
        return len(self.clauses) > 1

    @property
    def primary_intent(self) -> str:
        return self.clauses[0].intent if self.clauses else "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw": self.raw,
            "normalized": self.normalized,
            "clauses": [c.to_dict() for c in self.clauses],
        }


def _starts_with_verb(text: str) -> Optional[str]:
    """Return the matching verb if text begins with a known verb phrase."""
    lower = text.lower().strip()
    for verb in ALL_VERB_PREFIXES:
        if lower.startswith(verb + " ") or lower == verb:
            return verb
    return None


class SemanticParser:
    """Verb-frame semantic parser producing structured clauses for planning."""

    def parse(self, text: str) -> ParseResult:
        if not text or not text.strip():
            return ParseResult(raw=text or "", normalized="", clauses=[])

        raw = text.strip()
        normalized = self._normalize(raw)
        segments = self._segment(normalized)
        clauses: List[Clause] = []
        for seg in segments:
            clause = self._clause_from_segment(seg)
            clauses.extend(self._distribute_lists(clause))
        if not clauses:
            clauses = [Clause(raw=normalized, intent="unknown", target=normalized)]

        result = ParseResult(raw=raw, normalized=normalized, clauses=clauses)
        if len(clauses) > 1:
            clauses[-1].is_terminal = True
            logger.info(
                "SemanticParser: compound utterance (%d clauses): %s",
                len(clauses),
                [f"{c.intent}({c.target or c.arguments})" for c in clauses],
            )
        return result

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize(self, text: str) -> str:
        clean = text.strip()

        # Strip leading politeness patterns
        for pat in LEADING_POLITENESS:
            clean = re.sub(pat, "", clean, flags=re.IGNORECASE).strip()

        # Remove filler words
        for filler in FILLER_WORDS:
            clean = re.sub(rf"\b{re.escape(filler)}\b[,]?\s*", "", clean, flags=re.IGNORECASE)

        # Collapse whitespace and trailing punctuation (keep internal quotes)
        clean = re.sub(r"\s+", " ", clean).strip()
        clean = re.sub(r"[.?!]+$", "", clean).strip()

        # Separate run-together verbs: "openchrome" -> "open chrome"
        lower = clean.lower()
        for verb in ("open", "launch", "start", "close", "quit", "type", "write", "press", "scroll", "search"):
            if lower.startswith(verb) and len(lower) > len(verb) and lower[len(verb)] not in (" ", "'"):
                clean = f"{verb} {clean[len(verb):]}"
                break

        return clean

    # ------------------------------------------------------------------
    # Verb-anchored clause segmentation
    # ------------------------------------------------------------------

    def _segment(self, text: str) -> List[str]:
        """
        Split text at coordination boundaries, but ONLY when the following
        segment begins with a verb phrase. Protects quoted payloads.
        """
        if not text:
            return []

        # Protect quoted strings from splitting
        quotes: List[Tuple[int, int]] = []
        for m in re.finditer(r'"([^"]*)"|\'([^\']*)\'', text):
            quotes.append((m.start(), m.end()))

        def _in_quotes(pos: int) -> bool:
            return any(s <= pos < e for s, e in quotes)

        segments: List[str] = []
        current_start = 0

        for m in CONJUNCTION_PATTERN.finditer(text):
            conj_start = m.start()
            if _in_quotes(conj_start):
                continue
            rest = text[m.end():].strip()
            if not rest:
                continue
            # Only split if the remainder begins a new verb phrase
            if _starts_with_verb(rest):
                segments.append(text[current_start:conj_start].strip())
                current_start = m.end()

        segments.append(text[current_start:].strip())
        # Strip trailing punctuation left by conjunction consumption
        cleaned = []
        for s in segments:
            s = s.strip().rstrip(",;").strip()
            if s:
                cleaned.append(s)
        return cleaned

    # ------------------------------------------------------------------
    # Verb frames: clause interpretation
    # ------------------------------------------------------------------

    def _clause_from_segment(self, segment: str) -> Clause:
        lower = segment.lower().strip()
        clause = Clause(raw=segment)

        # Extract quoted payload if present (for type / search / rename)
        quoted = None
        qm = re.search(r'"([^"]*)"|\'([^\']*)\'', segment)
        if qm:
            quoted = qm.group(1) if qm.group(1) is not None else qm.group(2)
            clause.quoted_payload = quoted
            # Remove quotes from the working copy
            segment_clean = segment[:qm.start()].strip() + " " + segment[qm.end():].strip()
            segment_clean = re.sub(r"\s+", " ", segment_clean).strip()
        else:
            segment_clean = segment
        lower_clean = segment_clean.lower().strip()

        # --- Verify action ------------------------------------------------------
        for verb in VERIFY_VERBS:
            if lower_clean.startswith(verb + " ") or lower_clean == verb:
                rest = segment_clean[len(verb):].strip()
                rest = re.sub(r"^(that\s+|the\s+)", "", rest, flags=re.IGNORECASE).strip()
                clause.intent = "verify_action"
                clause.verb = verb
                clause.target = rest
                clause.arguments = {"expectation": rest}
                return clause

        # --- Screenshot / screen query -------------------------------------
        if re.search(r"\bscreenshot\b|\bscreen ?shot\b|\bcapture (the )?screen\b|\bcapture\b", lower_clean):
            clause.intent = "take_screenshot"
            clause.verb = "screenshot"
            clause.target = "screen"
            return clause
        if re.search(r"\bwhat('s| is) on (my )?(screen|display)\b|\bread (the )?screen\b|\bdescribe (the )?screen\b|\bread this\b", lower_clean):
            clause.intent = "read_screen"
            clause.verb = "read"
            clause.target = "screen_text"
            return clause

        # --- Window actions --------------------------------------------------
        for word, action in WINDOW_ACTION_MAP.items():
            if lower_clean.startswith(word) or re.search(rf"\b{re.escape(word)}\b.*\bwindow\b", lower_clean):
                clause.intent = "window_action"
                clause.verb = word
                clause.arguments = {"action": action}
                clause.target = action
                return clause
        if re.search(r"\b(full ?screen|fullscreen)\b", lower_clean):
            clause.intent = "window_action"
            clause.arguments = {"action": "fullscreen"}
            clause.target = "fullscreen"
            return clause

        # --- Volume / media ---------------------------------------------------
        if re.search(r"\bset volume\b|\bvolume\b", lower_clean):
            match = re.search(r"(\d+)\s*(?:percent|%)?", lower_clean)
            if match:
                clause.arguments = {"level": f"{match.group(1)}%"}
            elif "up" in lower_clean or "raise" in lower_clean or "louder" in lower_clean:
                clause.arguments = {"level": "up"}
            elif "down" in lower_clean or "lower" in lower_clean or "quieter" in lower_clean:
                clause.arguments = {"level": "down"}
            else:
                clause.arguments = {"level": "50%"}
            clause.intent = "set_volume"
            clause.verb = "volume"
            clause.target = str(clause.arguments["level"])
            return clause
        if re.search(r"\b(un)?mute\b|\bsilence\b", lower_clean):
            clause.intent = "mute_volume"
            clause.arguments = {"state": "unmute" if re.search(r"\bunmute\b", lower_clean) else "mute"}
            clause.target = clause.arguments["state"]
            return clause
        if re.search(r"\bplay\b|\bpause\b|\bskip\b|\bnext track\b|\bprevious track\b|\bstop (the )?(music|playback|media)\b", lower_clean):
            if "next" in lower_clean:
                action = "next"
            elif "previous" in lower_clean or "last" in lower_clean:
                action = "previous"
            elif "pause" in lower_clean:
                action = "pause"
            elif "play" in lower_clean:
                action = "play"
            else:
                action = "play_pause"
            clause.intent = "media_control"
            clause.arguments = {"action": action}
            clause.target = action
            return clause

        # --- Scroll ------------------------------------------------------------
        if re.search(r"^scroll|^page (up|down)", lower_clean):
            direction = "down"
            if re.search(r"\bup\b", lower_clean):
                direction = "up"
            elif re.search(r"\bdown\b", lower_clean):
                direction = "down"
            count = 1
            if re.search(r"\b(twice|2 times|two times)\b", lower_clean):
                count = 2
            elif re.search(r"\b(three times|thrice)\b", lower_clean):
                count = 3
            else:
                nm = re.search(r"\b(\d+)\s+times\b", lower_clean)
                if nm:
                    count = max(1, min(int(nm.group(1)), 20))
            clause.intent = "scroll"
            clause.arguments = {"direction": direction, "count": count}
            clause.target = direction
            return clause

        # --- Wait -----------------------------------------------------------
        for verb in WAIT_VERBS:
            if lower_clean.startswith(verb):
                what = segment_clean[len(verb):].strip()
                clause.intent = "wait_action"
                clause.verb = verb
                clause.target = what
                clause.arguments = {"for": what}
                # "wait for the download" → also record a timeout
                tm = re.search(r"(\d+)\s*(seconds?|sec|s\b)", lower_clean)
                clause.arguments["timeout_sec"] = float(tm.group(1)) if tm else 30.0
                return clause

        # --- Press key --------------------------------------------------------
        for verb in PRESS_VERBS:
            if lower_clean.startswith(verb + " ") or lower_clean == verb:
                keyphrase = segment_clean[len(verb):].strip()
                keyphrase = re.sub(r"^(the\s+)?(key|button)\s+", "", keyphrase, flags=re.IGNORECASE).strip()
                # Do NOT steal "press the proceed button" → that's a click
                if keyphrase and not keyphrase.lower().endswith(("button", "option", "link", "tab", "field", "menu")):
                    key = self._parse_key(keyphrase)
                    if key:
                        clause.intent = "press_key"
                        clause.verb = verb
                        clause.arguments = {"key": key}
                        clause.target = key
                        return clause
                # Falls through to click handling below when it's a button

        # --- Click / select element ------------------------------------------
        for verb in CLICK_VERBS:
            if lower_clean.startswith(verb + " ") or lower_clean == verb:
                element = segment_clean[len(verb):].strip()
                element = re.sub(r"^(the\s+|a\s+|an\s+)", "", element, flags=re.IGNORECASE).strip()
                # "select 5x" → option click; "click the 5x option" → same
                option_m = re.match(r"^(.*?)\s+option$", element, flags=re.IGNORECASE)
                if option_m:
                    element = option_m.group(1).strip()
                # Strip trailing UI-role nouns: "the Proceed button" → "Proceed"
                role_m = re.match(r"^(.*?)\s+(button|link|tab|menu|menu item|field|checkbox|check box|toggle|switch|dropdown|drop ?down|radio button|item)$", element, flags=re.IGNORECASE)
                if role_m and role_m.group(1).strip():
                    element = role_m.group(1).strip()
                clause.intent = "click_element"
                clause.verb = verb
                clause.target = element
                clause.arguments = {"element": element}
                clause.is_referring = element.lower() in REFERRING_EXPRESSIONS
                return clause

        # --- Submit / proceed -------------------------------------------------
        for verb in SUBMIT_VERBS:
            if lower_clean == verb or lower_clean.startswith(verb + " ") or lower_clean.startswith(verb):
                clause.intent = "submit_action"
                clause.verb = verb
                clause.target = verb
                clause.arguments = {"action": verb}
                # "send" with a following target means send-to-someone (message) — treat as submit for now
                return clause

        # --- Type text ----------------------------------------------------------
        for verb in TYPE_VERBS:
            if lower_clean.startswith(verb + " ") or lower_clean == verb:
                payload = quoted
                if payload is None:
                    payload = segment_clean[len(verb):].strip()
                    payload = re.sub(r"^(this\s+|the\s+|a\s+)?(message|text|query|search( query)?)\s*", "", payload, flags=re.IGNORECASE).strip()
                if payload:
                    clause.intent = "type_text"
                    clause.verb = verb
                    clause.arguments = {"text": payload}
                    clause.target = payload[:40]
                    return clause

        # --- Search --------------------------------------------------------------
        for verb in SEARCH_VERBS:
            if lower_clean.startswith(verb + " ") or lower_clean == verb:
                payload = quoted
                if payload is None:
                    payload = segment_clean[len(verb):].strip()
                # File-system queries ("find the largest files") are NOT web searches
                if re.search(r"\b(largest|biggest|smallest|newest|oldest|recent|duplicate)\b", payload, re.IGNORECASE) and re.search(r"\bfiles?\b", payload, re.IGNORECASE):
                    size_m = re.search(r"\b(largest|biggest|smallest|newest|oldest|recent)\b", payload, re.IGNORECASE)
                    criterion = (size_m.group(1) or "").lower() if size_m else ""
                    clause.intent = "file_find"
                    clause.verb = verb
                    clause.target = payload
                    clause.arguments = {"criterion": criterion, "pattern": ""}
                    return clause
                engine = "google"
                if re.match(r"^youtube\b", verb) or "youtube" in lower_clean[:20]:
                    engine = "youtube"
                elif "duckduckgo" in lower_clean[:30]:
                    engine = "duckduckgo"
                # "search hello on youtube" / "search youtube for hello"
                on_m = re.search(r"\bon\s+(youtube|google|duckduckgo)\b", payload, flags=re.IGNORECASE)
                if on_m:
                    engine = on_m.group(1).lower()
                    payload = (payload[:on_m.start()] + payload[on_m.end():]).strip()
                ytf = re.match(r"^youtube\s+for\s+(.*)$", payload, flags=re.IGNORECASE)
                if ytf:
                    engine = "youtube"
                    payload = ytf.group(1).strip()
                if payload:
                    clause.intent = "browser_search"
                    clause.verb = verb
                    clause.arguments = {"query": payload, "engine": engine}
                    clause.target = payload
                    return clause

        # --- Download --------------------------------------------------------------
        for verb in DOWNLOAD_VERBS:
            if lower_clean.startswith(verb + " "):
                what = segment_clean[len(verb):].strip()
                clause.intent = "download_action"
                clause.verb = verb
                clause.target = what
                clause.arguments = {"what": what}
                clause.is_referring = what.lower() in REFERRING_EXPRESSIONS
                return clause

        # --- Save / convert --------------------------------------------------------
        for verb in SAVE_VERBS:
            if lower_clean.startswith(verb):
                rest = segment_clean[len(verb):].strip()
                rest = re.sub(r"^(it|this|that|them)\s+", "", rest, flags=re.IGNORECASE).strip()
                # If the verb already consumed "as" ("save as X", "convert to X"),
                # the remainder begins with the format itself
                as_consumed = verb.endswith(" as") or verb.endswith(" to")
                if as_consumed:
                    rest = re.sub(r"^(?:as|to)\s+", "", rest, flags=re.IGNORECASE).strip()
                fmt = None
                loc = None
                fmt_m = re.match(r"^(\w+)(?:\s+(?:in|to|into|at)\s+(.+))?$", rest, flags=re.IGNORECASE) if as_consumed else None
                if fmt_m:
                    fmt = fmt_m.group(1)
                    loc = fmt_m.group(2)
                else:
                    fmt_m2 = re.search(r"\bas\s+(\w+)\b", rest, flags=re.IGNORECASE)
                    if fmt_m2:
                        fmt = fmt_m2.group(1)
                    loc_m = re.search(r"\b(?:in|to|into|at)\s+(?:my\s+|the\s+)?(.+)$", rest, flags=re.IGNORECASE)
                    if loc_m:
                        loc = loc_m.group(1)
                if loc and loc.endswith(" folder"):
                    loc = loc[: -len(" folder")].strip()
                clause.intent = "save_action"
                clause.verb = verb
                clause.target = fmt or "file"
                clause.arguments = {"format": fmt, "location": loc}
                clause.is_referring = True  # saving refers to whatever is being worked on
                return clause

        # --- Fill form -------------------------------------------------------
        for verb in FILL_VERBS:
            if lower_clean.startswith(verb + " "):
                rest = segment_clean[len(verb):].strip()
                clause.intent = "fill_form"
                clause.verb = verb
                clause.target = rest or "form"
                clause.arguments = {"form": rest or "form"}
                clause.is_referring = True
                return clause

        # --- Generic UI interaction verbs (add/import/resize/...) -------------
        # These are resolved via perception: AT-SPI element match or vision.
        for verb in INTERACTION_VERBS:
            if lower_clean.startswith(verb + " "):
                rest = segment_clean[len(verb):].strip()
                clause.intent = "ui_action"
                clause.verb = verb
                clause.target = rest
                clause.arguments = {"action": verb, "object": rest}
                clause.is_referring = rest.lower() in REFERRING_EXPRESSIONS
                return clause

        # --- File operations ----------------------------------------------------------
        for verb in MOVE_VERBS + COPY_VERBS:
            if lower_clean.startswith(verb + " "):
                rest = segment_clean[len(verb):].strip()
                src, dst = self._split_src_dest(rest)
                clause.intent = "file_move" if verb in MOVE_VERBS else "file_copy"
                clause.verb = verb
                clause.target = src
                clause.arguments = {"source": src, "destination": dst}
                clause.is_referring = src.lower() in REFERRING_EXPRESSIONS
                return clause

        for verb in RENAME_VERBS:
            if lower_clean.startswith(verb):
                rest = segment_clean[len(verb):].strip()
                # Keep referring expressions intact for the resolver
                # ("rename this file to X" → source="this file")
                rest = re.sub(r"^(?:it|that|them)\s+", "", rest, flags=re.IGNORECASE).strip()
                parts = re.split(r"\bto\b|\bas\b", rest, maxsplit=1, flags=re.IGNORECASE)
                src = parts[0].strip() if parts else ""
                new_name = parts[1].strip() if len(parts) > 1 else quoted or ""
                if not src:
                    src = "this file"
                clause.intent = "file_rename"
                clause.verb = verb
                clause.target = src
                clause.arguments = {"source": src, "new_name": new_name}
                clause.is_referring = src.lower() in REFERRING_EXPRESSIONS or bool(re.match(r"^(this|that|the)\s", src.lower()))
                return clause

        for verb in DELETE_VERBS:
            if lower_clean.startswith(verb + " "):
                rest = segment_clean[len(verb):].strip()
                clause.intent = "file_delete"
                clause.verb = verb
                clause.target = rest
                clause.arguments = {"target": rest}
                clause.is_referring = rest.lower() in REFERRING_EXPRESSIONS
                return clause

        for verb in CREATE_VERBS:
            if lower_clean.startswith(verb):
                rest = segment_clean[len(verb):].strip()
                folder_m = re.match(r"^(?:a\s+|an\s+|new\s+)?(folder|directory|dir)\s+(?:called\s+|named\s+)?(.+)$", rest, flags=re.IGNORECASE)
                if folder_m:
                    clause.intent = "file_create_folder"
                    clause.verb = verb
                    clause.target = folder_m.group(2).strip()
                    clause.arguments = {"name": folder_m.group(2).strip()}
                    return clause
                file_m = re.match(r"^(?:a\s+|an\s+|new\s+)?(file|document)\s+(?:called\s+|named\s+)?(.+)$", rest, flags=re.IGNORECASE)
                if file_m:
                    clause.intent = "file_create_file"
                    clause.verb = verb
                    clause.target = file_m.group(2).strip()
                    clause.arguments = {"name": file_m.group(2).strip()}
                    return clause
                # "create folder X" without article
                bare_m = re.match(r"^(folder|directory)\s+(.+)$", rest, flags=re.IGNORECASE)
                if bare_m:
                    clause.intent = "file_create_folder"
                    clause.arguments = {"name": bare_m.group(2).strip()}
                    clause.target = bare_m.group(2).strip()
                    return clause
                break  # fall through if not folder/file

        for verb in COMPRESS_VERBS:
            if lower_clean.startswith(verb + " "):
                rest = segment_clean[len(verb):].strip()
                clause.intent = "file_compress"
                clause.verb = verb
                clause.target = rest
                clause.arguments = {"targets": rest}
                clause.is_referring = rest.lower() in REFERRING_EXPRESSIONS
                return clause

        for verb in EXTRACT_VERBS:
            if lower_clean.startswith(verb + " "):
                rest = segment_clean[len(verb):].strip()
                clause.intent = "file_extract"
                clause.verb = verb
                clause.target = rest
                clause.arguments = {"target": rest}
                return clause

        # --- Browser navigate ("go to X", "visit X") ------------------------------------
        for verb in NAVIGATE_VERBS + SWITCH_VERBS:
            if lower_clean.startswith(verb + " ") or lower_clean == verb:
                target = segment_clean[len(verb):].strip()
                target = re.sub(r"^(the\s+|my\s+)", "", target, flags=re.IGNORECASE).strip()
                if not target:
                    continue
                # URL or known site → navigate
                if self._looks_like_site(target):
                    clause.intent = "browser_navigate"
                    clause.verb = verb
                    clause.arguments = {"url": self._normalize_site_url(target)}
                    clause.target = self._normalize_site_url(target)
                    return clause
                # otherwise app/window switching
                clause.intent = "switch_window"
                clause.verb = verb
                clause.target = target
                clause.arguments = {"query": target}
                return clause

        # --- Close application ------------------------------------------------------------
        for verb in CLOSE_VERBS:
            if lower_clean.startswith(verb + " ") or lower_clean == verb:
                target = segment_clean[len(verb):].strip()
                target = re.sub(r"^(the\s+|my\s+)?(app|application|program|window)\s+", "", target, flags=re.IGNORECASE).strip()
                clause.intent = "close_application"
                clause.verb = verb
                if not target or target.lower() in ("this", "it", "that", "window", "this window", "current window"):
                    clause.arguments = {}
                    clause.target = "active_window"
                else:
                    clause.arguments = {"name": target}
                    clause.target = target
                return clause

        # --- Open application (default for "open X") ---------------------------------------
        for verb in OPEN_VERBS:
            if lower_clean.startswith(verb + " ") or lower_clean == verb:
                target = segment_clean[len(verb):].strip()
                target = re.sub(r"^(the\s+|my\s+|up\s+)", "", target, flags=re.IGNORECASE).strip()
                if not target:
                    continue
                # URL → navigate instead
                if self._looks_like_site(target):
                    clause.intent = "browser_navigate"
                    clause.verb = verb
                    url = self._normalize_site_url(target)
                    clause.arguments = {"url": url}
                    clause.target = url
                    return clause
                # Folder reference → open folder
                folder = self._folder_match(target)
                if folder:
                    clause.intent = "open_application"
                    clause.verb = verb
                    clause.arguments = {"name": target, "is_folder": True, "folder": folder}
                    clause.target = target
                    return clause
                # Referring expressions: "open this image" / "open this game" / "open it"
                ref = self._referring_kind(target)
                if ref == "file":
                    clause.intent = "file_open"
                    clause.verb = verb
                    clause.arguments = {"target": target}
                    clause.target = target
                    clause.is_referring = True
                    return clause
                if target.lower() in ("it", "this", "that", "them", "the file", "the image"):
                    # Bare pronoun: whatever was last worked on (file by default —
                    # the executor's context resolves it)
                    clause.intent = "file_open"
                    clause.verb = verb
                    clause.arguments = {"target": "this file"}
                    clause.target = "this file"
                    clause.is_referring = True
                    return clause
                # Search-result reference: "open the result", "open the first
                # result", "open the top video" — clicks a result link in the
                # current page (the executor supplies search context).
                res_m = re.match(
                    r"^(?:the\s+)?(?:first\s+|top\s+|second\s+|third\s+|next\s+|latest\s+|newest\s+)?(result|results|search result|video|link|item|game)\b",
                    target, flags=re.IGNORECASE,
                )
                if res_m and not self._referring_kind(target):
                    ordinal = "first"
                    for cand in ("second", "third"):
                        if re.search(rf"\b{cand}\b", target, flags=re.IGNORECASE):
                            ordinal = cand
                    clause.intent = "click_element"
                    clause.verb = verb
                    clause.arguments = {"element": "search result", "ordinal": ordinal}
                    clause.target = "search result"
                    clause.is_referring = True
                    return clause
                if ref == "content":
                    # Content inside the currently open application (game in Steam, channel in browser)
                    clause.intent = "ui_action"
                    clause.verb = verb
                    clause.arguments = {"action": "open", "object": target}
                    clause.target = target
                    clause.is_referring = True
                    return clause
                clause.intent = "open_application"
                clause.verb = verb
                clause.arguments = {"name": target}
                clause.target = target
                clause.is_referring = self._is_referring_phrase(target)
                return clause

        # --- Fallback: unknown segment -----------------------------------------------------
        clause.intent = "unknown"
        clause.target = segment_clean
        clause.confidence = 0.3
        return clause

    # ------------------------------------------------------------------
    # Coordination distribution: "open chrome and opera" → two clauses
    # ------------------------------------------------------------------

    def _distribute_lists(self, clause: Clause) -> List[Clause]:
        """
        For app-intent clauses whose target is a coordinated noun list
        ("chrome and opera", "chrome, steam and dolphin"), distribute the
        verb over every item. Only applies to intents where a bare noun
        phrase is a valid target (open/close/switch). Never applies to
        query/type payloads where 'and' is part of the content.
        """
        distributable = {
            "open_application", "close_application", "switch_window",
            "browser_navigate",
        }
        if clause.intent not in distributable:
            return [clause]
        if clause.quoted_payload is not None:
            return [clause]
        if clause.is_referring or clause.arguments.get("is_folder"):
            return [clause]

        target = clause.target
        if not target:
            return [clause]

        # Split on coordinating commas / "and" / "or" / "plus" / "also"
        parts = re.split(r"\s*(?:,|\band\b|\bor\b|\bplus\b|\balso\b)\s*", target, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) <= 1:
            return [clause]

        # NOTE: we deliberately do NOT re-check parts for verb phrases here.
        # Verb-anchored segmentation (upstream) already split any genuine
        # coordinated action ("open chrome and search hello"). What remains
        # after that split are coordinated TARGETS ("Google Chrome and opera"),
        # even when a target's first word happens to also be a verb
        # ("google" in "Google Chrome") — app names must never be mistaken
        # for verb phrases during distribution.

        clauses: List[Clause] = []
        for p in parts:
            sub = Clause(
                raw=f"{clause.verb} {p}",
                intent=clause.intent,
                verb=clause.verb,
                target=p,
                arguments=dict(clause.arguments),
                confidence=clause.confidence,
            )
            if clause.intent == "open_application":
                sub.arguments["name"] = p
                folder = self._folder_match(p)
                if folder:
                    sub.arguments["is_folder"] = True
                    sub.arguments["folder"] = folder
                elif self._looks_like_site(p):
                    sub.intent = "browser_navigate"
                    sub.arguments = {"url": self._normalize_site_url(p)}
            elif clause.intent == "browser_navigate":
                sub.arguments = {"url": self._normalize_site_url(p)}
            elif clause.intent == "close_application":
                sub.arguments = {"name": p}
            elif clause.intent == "switch_window":
                sub.arguments = {"query": p}
            clauses.append(sub)

        logger.info(
            "Distributed coordinated list '%s' into %d clauses (verb inheritance: %s)",
            target, len(clauses), clause.verb,
        )
        return clauses

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _referring_kind(self, target: str) -> Optional[str]:
        """Classify a referring target: file-type, content-type, or None."""
        t = target.lower().strip()
        t = re.sub(r"^(this|that|the|these|those|my)\s+", "", t)
        t = re.sub(r"\s+(file|folder)$", "", t)
        if t in FILE_TYPE_NOUNS:
            return "file"
        if t in CONTENT_TYPE_NOUNS:
            return "content"
        return None

    def _is_referring_phrase(self, target: str) -> bool:
        t = target.lower().strip()
        if t in REFERRING_EXPRESSIONS:
            return True
        return bool(re.match(r"^(this|that|these|those)\s+\w+", t))

    def _looks_like_site(self, target: str) -> bool:
        t = target.lower().strip()
        t = re.sub(r"^(the\s+|my\s+)", "", t)
        if t in KNOWN_SITE_NAMES:
            return True
        if URL_PATTERN.match(t):
            return True
        # "youtube dot com" style
        if re.search(r"\s+dot\s+", t):
            return True
        # Bare known-site stems with words: "example dot com" handled above;
        # also treat "the <site> website" / "<site> website"
        if t.endswith(" website") or t.endswith(" site"):
            stem = t[: -len(" website")] if t.endswith(" website") else t[: -len(" site")]
            stem = re.sub(r"^(the\s+|my\s+)", "", stem).strip()
            if stem in KNOWN_SITE_NAMES:
                return True
        return False

    def _normalize_site_url(self, target: str) -> str:
        t = target.lower().strip()
        t = re.sub(r"^(the\s+|my\s+)", "", t)
        t = re.sub(r"\s+dot\s+", ".", t)
        if t in KNOWN_SITE_NAMES:
            return KNOWN_SITE_NAMES[t]
        if t.startswith("http://") or t.startswith("https://"):
            return t
        return f"https://{t}"

    def _folder_match(self, target: str) -> Optional[str]:
        t = target.lower().strip()
        t = re.sub(r"^(my|the)\s+", "", t)
        t = re.sub(r"\s+(folder|directory)$", "", t).strip()
        if t in FOLDER_WORDS:
            return FOLDER_WORDS[t]
        if t.endswith(" folder") or t.endswith(" directory"):
            return t[: -len(" folder")].strip()
        return None

    def _parse_key(self, keyphrase: str) -> Optional[str]:
        """Parse a key phrase (possibly a combo like 'control w') into a keysym combo string."""
        clean = keyphrase.lower().strip()
        clean = re.sub(r"^(the\s+)", "", clean)
        clean = re.sub(r"\s+key$", "", clean)
        if not clean:
            return None
        tokens = re.split(r"[\s+\-]+", clean)
        mapped: List[str] = []
        for tok in tokens:
            if not tok:
                continue
            if tok in KEY_NAME_MAP:
                mapped.append(KEY_NAME_MAP[tok])
            elif len(tok) == 1 and tok.isalnum():
                mapped.append(tok.upper() if not tok.isdigit() else tok)
            else:
                return None
        if not mapped:
            return None
        # Single key vs combo (modifiers first)
        return "+".join(mapped)

    def _split_src_dest(self, rest: str) -> Tuple[str, str]:
        """Split 'X to Y' / 'X into Y' into (source, destination)."""
        parts = re.split(r"\b(?:to|into|in)\b", rest, maxsplit=1, flags=re.IGNORECASE)
        src = parts[0].strip()
        dst = parts[1].strip() if len(parts) > 1 else ""
        dst = re.sub(r"^(the\s+|my\s+)", "", dst, flags=re.IGNORECASE).strip()
        if dst.endswith(" folder"):
            dst = dst[: -len(" folder")].strip()
        return src, dst


# Module-level singleton for cheap access
_PARSER: Optional[SemanticParser] = None


def get_parser() -> SemanticParser:
    global _PARSER
    if _PARSER is None:
        _PARSER = SemanticParser()
    return _PARSER
