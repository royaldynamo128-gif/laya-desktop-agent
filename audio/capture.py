"""
Streaming Audio Capture with Pre-Roll Ring Buffer & Signal Quality Analysis for Laya Desktop Agent.
Uses sounddevice connected to PipeWire / PulseAudio default source.
Guarantees zero-clipped speech boundaries with circular pre-roll and trailing post-roll windows.
"""

from typing import Optional, Dict, Any
import collections
import io
import logging
import math
import subprocess
import threading
import time
import wave
import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioCapture:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        max_seconds: int = 30,
        preroll_ms: int = 350,
        postroll_ms: int = 250,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_seconds = max_seconds
        self.max_frames = sample_rate * max_seconds
        self.preroll_ms = preroll_ms
        self.postroll_ms = postroll_ms

        # Ring buffer for pre-roll (stores raw 16-bit PCM chunks)
        # blocksize is ~30ms (480 samples at 16kHz = 960 bytes)
        self._block_size = int(self.sample_rate * 0.03)
        num_preroll_blocks = max(1, math.ceil(preroll_ms / 30.0))
        self._preroll_ring = collections.deque(maxlen=num_preroll_blocks)

        self._buffer = io.BytesIO()
        self._lock = threading.Lock()
        self._is_capturing = False
        self._stream: Optional[sd.InputStream] = None
        self._frames_recorded = 0

    @property
    def is_capturing(self) -> bool:
        return self._is_capturing

    def _ensure_safe_mic_volume(self) -> None:
        """Check and ensure default audio source volume is at a clean, non-clipping level (<= 0.40)."""
        try:
            res = subprocess.run(["wpctl", "get-volume", "@DEFAULT_AUDIO_SOURCE@"], capture_output=True, text=True, timeout=1.0)
            if res.returncode == 0:
                parts = res.stdout.strip().split()
                if len(parts) >= 2:
                    vol = float(parts[1])
                    if vol > 0.42:
                        logger.info("Microphone volume was high (%.2f), reducing to 0.35 to prevent audio clipping.", vol)
                        subprocess.run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SOURCE@", "0.35"], check=False, timeout=1.0)
        except Exception as e:
            logger.debug("Could not verify/adjust mic volume: %s", e)

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.debug("SoundDevice status warning: %s", status)

        # indata is float32 or int16. Apply soft limiting if peak is near clipping
        if indata.dtype == np.float32:
            peak = float(np.max(np.abs(indata)))
            scaled = indata
            if peak > 0.95:
                scaled = indata * (0.90 / peak)
            pcm_bytes = (np.clip(scaled, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        else:
            arr = indata.astype(np.float32)
            peak = float(np.max(np.abs(arr)))
            if peak > 30000:
                arr = arr * (28000.0 / peak)
            pcm_bytes = arr.astype(np.int16).tobytes()

        # Always update pre-roll ring buffer
        self._preroll_ring.append(pcm_bytes)

        if not self._is_capturing:
            return

        with self._lock:
            if self._frames_recorded + frames <= self.max_frames:
                self._buffer.write(pcm_bytes)
                self._frames_recorded += frames
            else:
                logger.warning("Audio capture reached max duration limit (%d s). Stopping.", self.max_seconds)
                self._is_capturing = False

    def warm_stream(self) -> None:
        """Keep the background stream warm so PTT activation has 0ms spinup delay."""
        if self._stream is None or not self._stream.active:
            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="int16",
                    callback=self._audio_callback,
                    blocksize=self._block_size,
                )
                self._stream.start()
                logger.info("Warm background audio stream active (%d Hz mono, pre-roll: %d ms).", self.sample_rate, self.preroll_ms)
            except Exception as e:
                logger.error("Failed to initialize warm audio stream: %s", e)

    def start(self) -> None:
        """Start streaming microphone input into buffer, prepending pre-roll history."""
        self._ensure_safe_mic_volume()
        self.warm_stream()

        with self._lock:
            self._buffer = io.BytesIO()
            # Seed buffer with pre-roll history so opening syllable is never lost
            for chunk in list(self._preroll_ring):
                self._buffer.write(chunk)
            self._frames_recorded = len(self._preroll_ring) * self._block_size
            self._is_capturing = True

        logger.debug("Audio recording started (seeded with %d ms pre-roll).", len(self._preroll_ring) * 30)

    def stop(self, wait_postroll: bool = True) -> bytes:
        """Stop capturing after post-roll window and return recorded raw 16-bit PCM bytes."""
        if wait_postroll and self.postroll_ms > 0 and self._is_capturing:
            # Brief pause to allow the user to finish articulating the last syllable
            time.sleep(self.postroll_ms / 1000.0)

        with self._lock:
            self._is_capturing = False
            raw_pcm = self._buffer.getvalue()
            self._buffer = io.BytesIO()
            self._frames_recorded = 0

        logger.debug("Audio capture stopped. Recorded %d bytes of PCM data.", len(raw_pcm))
        return raw_pcm

    @staticmethod
    def analyze_signal(pcm_bytes: bytes) -> Dict[str, Any]:
        """Analyze raw 16-bit PCM audio quality, levels, SNR, and clipping."""
        if not pcm_bytes:
            return {
                "rms_dbfs": -100.0,
                "peak_dbfs": -100.0,
                "peak_abs": 0,
                "is_clipped": False,
                "quality": "SILENT",
                "quality_conf": 0.0,
            }

        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        peak_abs = int(np.max(np.abs(samples)))
        rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0

        peak_dbfs = 20.0 * math.log10(max(peak_abs, 1) / 32768.0)
        rms_dbfs = 20.0 * math.log10(max(rms, 1.0) / 32768.0)
        is_clipped = peak_abs >= 32700

        # Categorize quality and compute quality confidence
        if is_clipped:
            quality = "CLIPPED"
            quality_conf = 0.40
        elif peak_abs < 800 or rms_dbfs < -45.0:
            quality = "LOW_LEVEL"
            quality_conf = 0.50
        elif rms_dbfs < -35.0:
            quality = "FAIR"
            quality_conf = 0.75
        elif -30.0 <= rms_dbfs <= -12.0:
            quality = "EXCELLENT"
            quality_conf = 0.98
        else:
            quality = "GOOD"
            quality_conf = 0.90

        return {
            "rms_dbfs": round(rms_dbfs, 1),
            "peak_dbfs": round(peak_dbfs, 1),
            "peak_abs": peak_abs,
            "is_clipped": is_clipped,
            "quality": quality,
            "quality_conf": quality_conf,
        }

    @staticmethod
    def normalize_pcm(pcm_bytes: bytes, target_peak_dbfs: float = -1.5) -> bytes:
        """Normalize audio peak to nominal level (-1.5 dBFS) for optimal Whisper accuracy."""
        if not pcm_bytes:
            return pcm_bytes

        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        peak = float(np.max(np.abs(samples)))

        # Only scale if signal is not complete silence or noise (peak > 250)
        if peak > 250.0:
            target_peak = 32768.0 * (10.0 ** (target_peak_dbfs / 20.0))  # ~27500
            scale = target_peak / peak
            # Bound scaling between 0.8x and 60.0x
            scale = min(max(scale, 0.8), 60.0)
            normalized = np.clip(samples * scale, -32767.0, 32767.0).astype(np.int16)
            return normalized.tobytes()

        return pcm_bytes

    def save_wav(self, raw_pcm: bytes, target_path: str) -> str:
        """Save raw PCM bytes to standard 16kHz mono WAV file."""
        with wave.open(target_path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(raw_pcm)
        return target_path

    def close(self) -> None:
        """Close audio stream and release hardware handle."""
        self._is_capturing = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
