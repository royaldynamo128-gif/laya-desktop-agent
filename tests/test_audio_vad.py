"""
Unit Tests for Audio VAD and Silence Trimming.
"""

import numpy as np
from audio.vad import VoiceActivityDetector, VADResult


def test_vad_silence_detection():
    vad = VoiceActivityDetector(sample_rate=16000, aggressiveness=3)
    # Pure silence (1 second)
    silence = np.zeros(16000, dtype=np.int16).tobytes()
    res = vad.process(silence)
    assert res.has_speech is False
    assert res.speech_ratio == 0.0


def test_vad_tone_detection_and_trimming():
    vad = VoiceActivityDetector(sample_rate=16000, aggressiveness=2)
    sr = 16000
    # 0.5s silence + 0.5s 440Hz tone + 0.5s silence = 1.5s total
    silence1 = np.zeros(int(sr * 0.5), dtype=np.int16)
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 25000).astype(np.int16)
    silence2 = np.zeros(int(sr * 0.5), dtype=np.int16)

    pcm = np.concatenate([silence1, tone, silence2]).tobytes()
    res = vad.process(pcm)

    assert res.has_speech is True
    assert res.trimmed_duration_sec < res.original_duration_sec
