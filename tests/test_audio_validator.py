"""
Unit tests for AudioValidator and preprocessing module.
"""

import numpy as np
import pytest
from audio.validator import AudioValidator, AudioValidationResult


def test_audio_validator_empty():
    validator = AudioValidator()
    res = validator.validate_and_preprocess(b"")
    assert res.is_usable is False
    assert "No audio bytes" in res.failure_reason


def test_audio_validator_too_short():
    validator = AudioValidator(min_duration_sec=0.25)
    # 0.1s of audio = 1600 samples = 3200 bytes
    short_pcm = np.zeros(1600, dtype=np.int16).tobytes()
    res = validator.validate_and_preprocess(short_pcm)
    assert res.is_usable is False
    assert "too brief" in res.failure_reason.lower()


def test_audio_validator_silence():
    validator = AudioValidator()
    # 1.0s of pure silence
    silence = np.zeros(16000, dtype=np.int16).tobytes()
    res = validator.validate_and_preprocess(silence)
    assert res.is_usable is False
    assert "silent" in res.failure_reason.lower() or "muted" in res.failure_reason.lower()


def test_audio_validator_valid_tone():
    validator = AudioValidator()
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # 440 Hz tone with modulation (simulating speech envelope)
    envelope = np.sin(np.pi * t)
    carrier = np.sin(2 * np.pi * 440 * t)
    signal = (envelope * carrier * 15000).astype(np.int16)

    res = validator.validate_and_preprocess(signal.tobytes())
    assert res.is_usable is True
    assert res.failure_reason == ""
    assert res.preprocessed_pcm is not None
    assert len(res.preprocessed_pcm) == len(signal.tobytes())

    # Preprocessed audio peak should be close to -1.5 dBFS (~27500)
    samples = np.frombuffer(res.preprocessed_pcm, dtype=np.int16)
    peak = np.max(np.abs(samples))
    assert 22000 <= peak <= 32767


def test_audio_validator_dc_offset_removal():
    validator = AudioValidator()
    sr = 16000
    # Tone with severe DC offset (+5000)
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    tone = (np.sin(2 * np.pi * 300 * t) * 8000 + 5000).astype(np.float32)

    cleaned = validator.remove_dc_and_rumble(tone)
    mean_dc = float(np.mean(cleaned))
    assert abs(mean_dc) < 50.0  # DC offset eliminated
