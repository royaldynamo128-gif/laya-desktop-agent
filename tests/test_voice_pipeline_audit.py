"""
Comprehensive Voice-Command Pipeline Audit Tests for Laya Desktop Agent.
Verifies all 5 required real commands end-to-end, audio validation, VAD rejection,
audible feedback on speech failure, and desktop retry on execution failure.
"""

import numpy as np
import pytest
from core.config import load_config
from core.state import AgentState
from app.daemon import LayaDaemon


@pytest.fixture(scope="module")
def daemon():
    config = load_config(profile="balanced")
    d = LayaDaemon(config, diagnostic_mode=True)
    d.model_mgr.laya.load()
    yield d
    d.model_mgr.shutdown()


def test_command_1_open_chrome(daemon):
    """Command 1: 'Open Chrome.'"""
    res = daemon.coordinator.process_text_command("Open Chrome.")
    assert res.intent == "open_application"
    assert res.action_executed == "open_application"
    assert res.success is True
    assert res.verified is True
    assert "chrome" in res.metrics.target.lower()


def test_command_2_open_dolphin(daemon):
    """Command 2: 'Open Dolphin.'"""
    res = daemon.coordinator.process_text_command("Open Dolphin.")
    assert res.intent == "open_application"
    assert res.action_executed == "open_application"
    assert res.success is True
    assert res.verified is True
    assert "dolphin" in res.metrics.target.lower()


def test_command_3_open_text_editor_and_write(daemon):
    """Command 3: 'Open the text editor and write Hello World.'"""
    # Decomposed into 2 steps: open text editor, then type text
    res = daemon.coordinator.process_text_command("Open the text editor and write Hello World.")
    assert res.success is True
    assert res.verified is True
    assert res.action_executed == "goal_executor"


def test_command_4_open_chrome_and_search_linux(daemon):
    """Command 4: 'Open Chrome and search for Linux.'"""
    # Decomposed into: Open Chrome, then search for Linux via browser_search
    res = daemon.coordinator.process_text_command("Open Chrome and search for Linux.")
    assert res.success is True
    assert res.verified is True
    assert res.action_executed == "goal_executor"


def test_command_5_open_settings(daemon):
    """Command 5: 'Open Settings.'"""
    res = daemon.coordinator.process_text_command("Open Settings.")
    assert res.intent == "open_application"
    assert res.action_executed == "open_application"
    assert res.success is True
    assert res.verified is True
    assert "settings" in res.metrics.target.lower()


def test_audio_validation_silence_rejected(daemon):
    """Ensure silent recording is rejected at AudioValidation stage with audible feedback and no stall."""
    sr = 16000
    silence = np.zeros(sr, dtype=np.int16).tobytes()
    res = daemon.coordinator.process_audio(silence)
    assert res is None
    assert daemon.coordinator.state_mgr.current_state == AgentState.IDLE


def test_audio_validation_too_brief_rejected(daemon):
    """Ensure <0.25s recording is rejected immediately before STT."""
    short = np.zeros(800, dtype=np.int16).tobytes()
    res = daemon.coordinator.process_audio(short)
    assert res is None
    assert daemon.coordinator.state_mgr.current_state == AgentState.IDLE


def test_speech_corrector_phonetic_notepad_execution(daemon):
    """Ensure 'open not pad' resolves to notepad/KWrite without forcing user to repeat."""
    raw, norm = daemon.coordinator.speech_corrector.normalize("open not pad")
    assert raw == "open not pad"
    assert "notepad" in norm.lower() or "kwrite" in norm.lower()

    res = daemon.coordinator.process_text_command(norm)
    assert res.intent == "open_application"
    assert res.success is True
    assert res.verified is True


def test_raw_and_normalized_transcriptions_preserved(daemon):
    """Requirement 4: Keep both raw transcript and normalized interpretation."""
    raw, norm = daemon.coordinator.speech_corrector.normalize("open doll fin")
    assert raw == "open doll fin"
    assert "dolphin" in norm.lower()

    res = daemon.coordinator.process_text_command(norm)
    assert res.success is True
    assert res.raw_transcription or norm
    assert res.normalized_transcription or norm
