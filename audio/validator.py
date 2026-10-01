"""
Audio Validation and Preprocessing Engine for Laya Desktop Agent.
Enforces strict audio quality, energy thresholds, DC offset removal,
rumble high-pass filtering, and dynamic range validation before STT.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioValidationResult:
    is_usable: bool
    failure_reason: str
    peak_dbfs: float
    rms_dbfs: float
    peak_abs: int
    dynamic_range_db: float
    duration_sec: float
    is_clipped: bool
    preprocessed_pcm: bytes
    metrics: Dict[str, Any] = field(default_factory=dict)


class AudioValidator:
    """Validates raw microphone PCM recordings and conditions audio for Faster-Whisper."""

    def __init__(
        self,
        sample_rate: int = 16000,
        min_duration_sec: float = 0.25,
        max_duration_sec: float = 30.0,
        min_rms_dbfs: float = -50.0,
        min_peak_abs: int = 350,
        min_dynamic_range_db: float = 3.5,
    ):
        self.sample_rate = sample_rate
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.min_rms_dbfs = min_rms_dbfs
        self.min_peak_abs = min_peak_abs
        self.min_dynamic_range_db = min_dynamic_range_db

    def validate_and_preprocess(self, pcm_bytes: bytes) -> AudioValidationResult:
        """Validate raw 16-bit PCM audio, remove DC offset/rumble, and normalize volume."""
        if not pcm_bytes:
            return AudioValidationResult(
                is_usable=False,
                failure_reason="No audio bytes captured",
                peak_dbfs=-100.0,
                rms_dbfs=-100.0,
                peak_abs=0,
                dynamic_range_db=0.0,
                duration_sec=0.0,
                is_clipped=False,
                preprocessed_pcm=b"",
                metrics={"quality": "EMPTY"},
            )

        duration_sec = len(pcm_bytes) / (self.sample_rate * 2)

        # 1. Duration validation
        if duration_sec < self.min_duration_sec:
            return AudioValidationResult(
                is_usable=False,
                failure_reason=f"Recording too brief ({duration_sec:.2f}s < {self.min_duration_sec:.2f}s)",
                peak_dbfs=-100.0,
                rms_dbfs=-100.0,
                peak_abs=0,
                dynamic_range_db=0.0,
                duration_sec=duration_sec,
                is_clipped=False,
                preprocessed_pcm=b"",
                metrics={"quality": "TOO_BRIEF"},
            )

        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        peak_abs = int(np.max(np.abs(samples))) if len(samples) > 0 else 0
        rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0

        peak_dbfs = 20.0 * math.log10(max(peak_abs, 1) / 32768.0)
        rms_dbfs = 20.0 * math.log10(max(rms, 1.0) / 32768.0)
        dynamic_range_db = peak_dbfs - rms_dbfs
        is_clipped = peak_abs >= 32700

        # 2. Silence / Muted mic check
        if rms_dbfs < self.min_rms_dbfs or peak_abs < self.min_peak_abs:
            logger.info(
                "Audio validation rejected: Silence/Muted (RMS=%.1f dBFS, Peak=%d)",
                rms_dbfs,
                peak_abs,
            )
            return AudioValidationResult(
                is_usable=False,
                failure_reason="Audio is silent or microphone is muted",
                peak_dbfs=round(peak_dbfs, 1),
                rms_dbfs=round(rms_dbfs, 1),
                peak_abs=peak_abs,
                dynamic_range_db=round(dynamic_range_db, 1),
                duration_sec=round(duration_sec, 2),
                is_clipped=is_clipped,
                preprocessed_pcm=b"",
                metrics={"quality": "SILENT"},
            )

        # 3. Dynamic range / ambient noise floor check
        if rms_dbfs < -40.0 and dynamic_range_db < self.min_dynamic_range_db:
            logger.info(
                "Audio validation rejected: Flat noise floor (RMS=%.1f dBFS, DynRange=%.1f dB)",
                rms_dbfs,
                dynamic_range_db,
            )
            return AudioValidationResult(
                is_usable=False,
                failure_reason="No speech dynamics detected (ambient noise floor)",
                peak_dbfs=round(peak_dbfs, 1),
                rms_dbfs=round(rms_dbfs, 1),
                peak_abs=peak_abs,
                dynamic_range_db=round(dynamic_range_db, 1),
                duration_sec=round(duration_sec, 2),
                is_clipped=is_clipped,
                preprocessed_pcm=b"",
                metrics={"quality": "NOISE_FLOOR"},
            )

        # 4. Audio Conditioning / Preprocessing
        # Step A: Remove DC offset and low frequency rumble (<100Hz)
        cleaned = self.remove_dc_and_rumble(samples)

        # Step B: Normalize peak to nominal level (-1.5 dBFS) for Whisper
        normalized = self.normalize_peak(cleaned, target_peak_dbfs=-1.5)

        preprocessed_bytes = normalized.astype(np.int16).tobytes()

        quality = "EXCELLENT" if rms_dbfs >= -30.0 else ("GOOD" if rms_dbfs >= -40.0 else "FAIR")
        if is_clipped:
            quality = "CLIPPED"

        return AudioValidationResult(
            is_usable=True,
            failure_reason="",
            peak_dbfs=round(peak_dbfs, 1),
            rms_dbfs=round(rms_dbfs, 1),
            peak_abs=peak_abs,
            dynamic_range_db=round(dynamic_range_db, 1),
            duration_sec=round(duration_sec, 2),
            is_clipped=is_clipped,
            preprocessed_pcm=preprocessed_bytes,
            metrics={"quality": quality, "snr_db": round(dynamic_range_db, 1)},
        )

    def remove_dc_and_rumble(self, samples: np.ndarray) -> np.ndarray:
        """Remove DC offset and low-frequency mechanical rumble via moving baseline subtraction."""
        if len(samples) < 160:
            return samples - np.mean(samples)

        # 1. Zero-center (remove DC offset)
        centered = samples - np.mean(samples)

        # 2. Moving average baseline subtraction (160 samples = 10ms = ~100Hz cutoff at 16kHz)
        kernel_size = 160
        kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
        baseline = np.convolve(centered, kernel, mode="same")
        filtered = centered - baseline
        return filtered

    def normalize_peak(self, samples: np.ndarray, target_peak_dbfs: float = -1.5) -> np.ndarray:
        """Scale audio peak to target dBFS with safety bounds."""
        peak = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
        if peak < 250.0:
            return samples

        target_peak = 32768.0 * (10.0 ** (target_peak_dbfs / 20.0))  # ~27500
        scale = target_peak / peak
        # Bound gain between 0.8x and 35.0x
        scale = min(max(scale, 0.8), 35.0)
        scaled = samples * scale
        return np.clip(scaled, -32767.0, 32767.0)
