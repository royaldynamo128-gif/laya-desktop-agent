"""
Speech Recognition Error Correction and Contextual Transcript Normalizer.
Resolves Whisper phonetic transcript slips, run-together tokens, filler words,
compound multi-action conjunctions, and contextually maps spoken phrases
to verified desktop applications and actions using desktop state and catalog.
"""

from typing import List, Optional, Set, Tuple, Any
import logging
import re
from integrations.app_catalog import AppCatalog

logger = logging.getLogger(__name__)

FILLER_WORDS_REGEX = re.compile(
    r"\b(?:um|uh|er|ah|like|you know|please|could you|can you)\b",
    re.IGNORECASE,
)

VERB_PREFIXES = ["open", "launch", "start", "close", "quit", "switch", "focus", "type", "write", "press"]

# Common phonetic STT slips
PHONETIC_SLIPS = [
    (re.compile(r"\b(?:not pad|note pad|notepadd)\b", re.IGNORECASE), "notepad"),
    (re.compile(r"\b(?:doll fin|dolfin|dolphen)\b", re.IGNORECASE), "Dolphin"),
    (re.compile(r"\b(?:crome|crowm)\b", re.IGNORECASE), "Chrome"),
    (re.compile(r"\b(?:set things|setings)\b", re.IGNORECASE), "Settings"),
    (re.compile(r"\bhello word\b", re.IGNORECASE), "Hello World"),
    (re.compile(r"\bwright\b", re.IGNORECASE), "write"),
]

CLAUSE_SPLIT_REGEX = re.compile(
    r"\b(?:and then|then|after that|and also|and)\b|;|, and|,",
    re.IGNORECASE,
)


class SpeechCorrector:
    """Corrects imperfect speech transcripts using desktop context and app catalog."""

    def __init__(self, catalog: Optional[AppCatalog] = None):
        self.catalog = catalog

    def normalize(
        self,
        transcript: str,
        desktop_state: Optional[Any] = None,
        history: Optional[List[str]] = None,
    ) -> Tuple[str, str]:
        """
        Produce both raw transcript and context-normalized interpretation.
        Returns: (raw_transcript, normalized_transcript)
        """
        if not transcript:
            return "", ""

        raw = transcript.strip()
        normalized = self.correct(raw, desktop_state=desktop_state, history=history)
        return raw, normalized

    def correct(
        self,
        transcript: str,
        desktop_state: Optional[Any] = None,
        history: Optional[List[str]] = None,
    ) -> str:
        """Apply full transcript cleaning, phonetic rectification, multi-clause splitting, and app repair."""
        if not transcript:
            return ""

        raw = transcript.strip()

        # 1. Strip filler words
        clean = FILLER_WORDS_REGEX.sub("", raw).strip()
        clean = re.sub(r"\s+", " ", clean).strip()

        # 2. Check for run-together verb + noun (e.g. "openfirefox" -> "open firefox")
        for verb in VERB_PREFIXES:
            if clean.lower().startswith(verb) and len(clean) > len(verb):
                # If there was no space between verb and remainder:
                if not clean.lower().startswith(verb + " "):
                    remainder = clean[len(verb):].strip()
                    clean = f"{verb} {remainder}"
                    logger.info("Separated run-together command: '%s' -> '%s'", raw, clean)
                    break

        # 3. Direct phonetic slip replacements
        for pattern, replacement in PHONETIC_SLIPS:
            if pattern.search(clean):
                old = clean
                clean = pattern.sub(replacement, clean)
                logger.info("Phonetic slip repaired: '%s' -> '%s'", old, clean)

        # 4. Multi-clause repair
        # If compound clauses exist (e.g. "Open Chrome and search for Linux" or "Open the text editor and write Hello World")
        # split on conjunctions, repair clauses individually, then rejoin
        if any(c in clean.lower() for c in [" and ", " then ", ";"]):
            clean = self._repair_compound_clauses(clean, desktop_state=desktop_state)
        else:
            # Single clause app catalog repair
            if self.catalog:
                clean = self._repair_app_names(clean)

        # 5. Desktop-contextual resolution
        if desktop_state and hasattr(desktop_state, "active_window_name"):
            clean = self._apply_desktop_context(clean, desktop_state)

        return clean

    def _repair_compound_clauses(self, text: str, desktop_state: Optional[Any] = None) -> str:
        """Split compound command, normalize each clause (e.g. app name, action), and reassemble."""
        # Find connectors and split
        match = re.search(r"\b(and then|then|after that|and also|and)\b", text, re.IGNORECASE)
        if not match:
            return text

        connector = match.group(0)
        idx = match.start()
        first_clause = text[:idx].strip()
        second_clause = text[idx + len(connector):].strip()

        # Repair first clause
        if self.catalog:
            first_clause = self._repair_app_names(first_clause)

        # Clean second clause
        for pattern, replacement in PHONETIC_SLIPS:
            second_clause = pattern.sub(replacement, second_clause)

        # Normalize second clause verbs if needed (e.g. "search Linux" -> "search for Linux")
        if second_clause.lower().startswith("search ") and not second_clause.lower().startswith("search for "):
            query = second_clause[7:].strip()
            second_clause = f"search for {query}"

        return f"{first_clause} {connector} {second_clause}".strip()

    def _repair_app_names(self, text: str) -> str:
        """Detect if command contains an application query and rectify phonetic mistranscriptions."""
        if not self.catalog:
            return text

        lower = text.lower().strip()

        # Identify target phrase after command verb
        for verb in ["open ", "launch ", "start ", "switch to ", "focus on ", "close "]:
            if lower.startswith(verb):
                target_phrase = text[len(verb):].strip()
                if not target_phrase:
                    continue

                # Query app catalog for phonetic or fuzzy match
                resolved = self.catalog.resolve(target_phrase)
                if resolved and resolved.confidence >= 0.80:
                    canonical_name = resolved.name
                    # If resolved name differs noticeably, substitute it
                    if canonical_name.lower() != target_phrase.lower():
                        logger.info(
                            "Corrected speech target: '%s' -> '%s' (conf: %.2f, tier: %s)",
                            target_phrase,
                            canonical_name,
                            resolved.confidence,
                            resolved.match_tier,
                        )
                        return f"{text[:len(verb)]}{canonical_name}"

        return text

    def _apply_desktop_context(self, text: str, desktop_state: Any) -> str:
        """Apply active window information to contextual pronouns."""
        lower = text.lower().strip()
        if lower in ("close this", "close it", "close window"):
            active = getattr(desktop_state, "active_window_name", "")
            if active:
                return f"close {active}"
        return text
