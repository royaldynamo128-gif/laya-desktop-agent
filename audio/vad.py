"""
Voice Activity Detection (VAD) for Laya Desktop Agent.
Uses WebRTC VAD to trim silence and detect speech boundaries with <1ms latency on CPU.
"""

from dataclasses import dataclass
from typing import List, Tuple
import logging
import webrtcvad

logger = logging.getLogger(__name__)


@dataclass
class VADResult:
    has_speech: bool
    speech_pcm: bytes
    speech_ratio: float
    original_duration_sec: float
    trimmed_duration_sec: float


class VoiceActivityDetector:
    def __init__(self, sample_rate: int = 16000, aggressiveness: int = 2, frame_duration_ms: int = 30):
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError(f"Unsupported sample rate for WebRTC VAD: {sample_rate}")
        if frame_duration_ms not in (10, 20, 30):
            raise ValueError(f"Unsupported frame duration: {frame_duration_ms}")

        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_bytes = int(sample_rate * (frame_duration_ms / 1000.0) * 2)  # 16-bit = 2 bytes/sample
        self.vad = webrtcvad.Vad(aggressiveness)

    def process(self, pcm_bytes: bytes, padding_frames: int = 8) -> VADResult:
        """Process raw 16-bit PCM bytes: detect speech and trim leading/trailing silence."""
        if not pcm_bytes or len(pcm_bytes) < self.frame_bytes:
            return VADResult(
                has_speech=False,
                speech_pcm=b"",
                speech_ratio=0.0,
                original_duration_sec=0.0,
                trimmed_duration_sec=0.0,
            )

        # Split into exact frame_bytes slices
        num_frames = len(pcm_bytes) // self.frame_bytes
        frames: List[Tuple[bytes, bool]] = []
        voiced_count = 0

        for i in range(num_frames):
            frame = pcm_bytes[i * self.frame_bytes : (i + 1) * self.frame_bytes]
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            if is_speech:
                voiced_count += 1
            frames.append((frame, is_speech))

        speech_ratio = voiced_count / num_frames if num_frames > 0 else 0.0
        original_duration = len(pcm_bytes) / (self.sample_rate * 2)

        # Require at least 3 voiced frames (~90ms) and min speech ratio
        if voiced_count < 3 or speech_ratio < 0.05:
            logger.debug("No usable speech detected (voiced frames: %d, ratio: %.2f)", voiced_count, speech_ratio)
            return VADResult(
                has_speech=False,
                speech_pcm=b"",
                speech_ratio=speech_ratio,
                original_duration_sec=original_duration,
                trimmed_duration_sec=0.0,
            )

        # Find first and last voiced frame indices
        first_voiced = 0
        for idx, (_, is_speech) in enumerate(frames):
            if is_speech:
                first_voiced = max(0, idx - padding_frames)
                break

        last_voiced = len(frames) - 1
        for idx in range(len(frames) - 1, -1, -1):
            if frames[idx][1]:
                last_voiced = min(len(frames) - 1, idx + padding_frames)
                break

        trimmed_frames = frames[first_voiced : last_voiced + 1]
        trimmed_pcm = b"".join(f[0] for f in trimmed_frames)
        trimmed_duration = len(trimmed_pcm) / (self.sample_rate * 2)

        logger.debug(
            "VAD trimmed audio: %.2fs -> %.2fs (speech ratio: %.2f)",
            original_duration,
            trimmed_duration,
            speech_ratio,
        )

        return VADResult(
            has_speech=True,
            speech_pcm=trimmed_pcm,
            speech_ratio=speech_ratio,
            original_duration_sec=original_duration,
            trimmed_duration_sec=trimmed_duration,
        )
