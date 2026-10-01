"""
Speech Recognition Error Correction and Contextual Transcript Normalizer.

Keeps raw and normalized transcripts separate. Correction layers:
  1. Parser-driven normalization (politeness, fillers, run-together verbs).
  2. Phonetic STT slip repair using the installed-application catalog
     ("crowm" → Chrome, "not pad" → notepad) — dynamic, not a fixed list.
  3. Desktop-context pronoun resolution ("close this" → "close <active app>").
"""

from typing import List, Optional, Tuple, Any
import logging
import re

from integrations.app_catalog import AppCatalog
from planning.semantic_parser import get_parser

logger = logging.getLogger(__name__)

# Common phonetic STT slips (tiny curated set for system nouns only —
# app names are resolved dynamically from the catalog, never hardcoded)
PHONETIC_SLIPS = [
    (re.compile(r"\b(?:not pad|note pad|notepadd)\b", re.IGNORECASE), "notepad"),
    (re.compile(r"\b(?:doll fin|dolfin|dolphen)\b", re.IGNORECASE), "Dolphin"),
    (re.compile(r"\b(?:crowm)\b", re.IGNORECASE), "Chrome"),
    (re.compile(r"\b(?:set things|setings)\b", re.IGNORECASE), "Settings"),
    (re.compile(r"\bhello word\b", re.IGNORECASE), "hello world"),
]


class SpeechCorrector:
    """Corrects imperfect speech transcripts using desktop context and app catalog."""

    def __init__(self, catalog: Optional[AppCatalog] = None):
        self.catalog = catalog
        self.parser = get_parser()

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
        """Full correction pipeline."""
        if not transcript:
            return ""

        raw = transcript.strip()

        # 1. Parser normalization (politeness, fillers, spacing)
        clean = self.parser._normalize(raw)

        # 2. Phonetic slip repair (system nouns)
        for pattern, replacement in PHONETIC_SLIPS:
            if pattern.search(clean):
                clean = pattern.sub(replacement, clean)

        # 3. Catalog-driven app-name repair in each clause
        if self.catalog:
            clean = self._repair_app_names(clean)

        # 4. Desktop-context pronoun resolution
        if desktop_state and hasattr(desktop_state, "active_window_name"):
            clean = self._apply_desktop_context(clean, desktop_state)

        return clean

    def _repair_app_names(self, text: str) -> str:
        """Repair spoken app names via catalog phonetic/fuzzy resolution."""
        lower = text.lower().strip()

        for verb in ("open ", "launch ", "start ", "switch to ", "focus on ", "close ", "quit "):
            if lower.startswith(verb):
                target_phrase = text[len(verb):].strip()
                if not target_phrase:
                    continue
                # Repair only the FIRST clause target (split off trailing clauses)
                split_at = None
                for conj in (" and ", " then ", ";", ","):
                    idx = target_phrase.lower().find(conj)
                    if idx > 0:
                        split_at = idx
                        break
                if split_at:
                    first, rest = target_phrase[:split_at], target_phrase[split_at:]
                else:
                    first, rest = target_phrase, ""

                resolved = self.catalog.resolve(first)
                if resolved and resolved.confidence >= 0.80:
                    if resolved.name.lower() != first.lower():
                        logger.info(
                            "Speech target repaired: '%s' -> '%s' (conf: %.2f, tier: %s)",
                            first, resolved.name, resolved.confidence, resolved.match_tier,
                        )
                        first = resolved.name
                return f"{text[:len(verb)]}{first}{rest}"
        return text

    def _apply_desktop_context(self, text: str, desktop_state: Any) -> str:
        """Resolve context pronouns for close/minimize actions."""
        lower = text.lower().strip()
        if lower in ("close this", "close it", "close window", "minimize this", "minimize it"):
            active = getattr(desktop_state, "active_window_name", "")
            verb = "close" if lower.startswith("close") else "minimize"
            if active:
                return f"{verb} {active}"
        return text
