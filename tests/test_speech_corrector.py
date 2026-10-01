"""
Unit tests for SpeechCorrector and transcript normalization.
"""

from dataclasses import dataclass
import pytest
from planning.speech_corrector import SpeechCorrector
from integrations.app_catalog import AppCatalog


@dataclass
class DummyDesktopObservation:
    active_window_name: str = "Google Chrome"
    active_window_class: str = "google-chrome"


def test_speech_corrector_filler_words():
    corrector = SpeechCorrector()
    res = corrector.correct("um please open chrome")
    assert "chrome" in res
    assert not res.startswith("um")
    assert "please" not in res


def test_speech_corrector_run_together():
    corrector = SpeechCorrector()
    res = corrector.correct("openfirefox")
    assert res == "open firefox"

    res2 = corrector.correct("launchdolphin")
    assert res2 == "launch dolphin"


def test_speech_corrector_phonetic_slips():
    catalog = AppCatalog()
    corrector = SpeechCorrector(catalog=catalog)

    # 1. "open not pad" -> "open notepad" / KWrite
    res = corrector.correct("open not pad")
    assert "notepad" in res.lower() or "kwrite" in res.lower()

    # 2. "open doll fin" -> "open Dolphin"
    res = corrector.correct("open doll fin")
    assert "dolphin" in res.lower()

    # 3. "open crome" -> "open Chrome"
    res = corrector.correct("open crome")
    assert "chrome" in res.lower()

    # 4. "open set things" -> "open Settings"
    res = corrector.correct("open set things")
    assert "settings" in res.lower()

    # 5. "open the text editor and write hello word"
    res = corrector.correct("open the text editor and write hello word")
    assert "hello world" in res.lower()
    assert "write" in res.lower()


def test_speech_corrector_multi_clause_compound():
    catalog = AppCatalog()
    corrector = SpeechCorrector(catalog=catalog)

    res = corrector.correct("Open Chrome and search for Linux")
    assert "chrome" in res.lower()
    assert "search for linux" in res.lower()

    res2 = corrector.correct("Open the text editor and write Hello World")
    assert "write hello world" in res2.lower()


def test_speech_corrector_normalize_tuple():
    catalog = AppCatalog()
    corrector = SpeechCorrector(catalog=catalog)

    raw, norm = corrector.normalize("open not pad")
    assert raw == "open not pad"
    assert "notepad" in norm.lower() or "kwrite" in norm.lower()


def test_speech_corrector_desktop_context():
    corrector = SpeechCorrector()
    obs = DummyDesktopObservation(active_window_name="Dolphin")
    res = corrector.correct("close this", desktop_state=obs)
    assert res == "close Dolphin"
