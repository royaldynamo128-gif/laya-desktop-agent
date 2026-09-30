"""
Streaming Audio Capture with Bounded Ring Buffer for Laya Desktop Agent.
Uses sounddevice connected to PipeWire / PulseAudio default source.
"""

from typing import Optional
import io
import logging
import threading
import time
import wave
import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioCapture:
    def __init__(self, sample_rate: int = 16000, channels: int = 1, max_seconds: int = 30):
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_seconds = max_seconds
        self.max_frames = sample_rate * max_seconds
        
        self._buffer = io.BytesIO()
        self._lock = threading.Lock()
        self._is_capturing = False
        self._stream: Optional[sd.InputStream] = None
        self._frames_recorded = 0

    @property
    def is_capturing(self) -> bool:
        return self._is_capturing

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.debug("SoundDevice status warning: %s", status)

        if not self._is_capturing:
            return

        # indata is float32 or int16. Convert to 16-bit PCM bytes
        if indata.dtype == np.float32:
            # Scale float [-1.0, 1.0] to int16
            pcm_data = (np.clip(indata, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        else:
            pcm_data = indata.astype(np.int16).tobytes()

        with self._lock:
            if self._frames_recorded + frames <= self.max_frames:
                self._buffer.write(pcm_data)
                self._frames_recorded += frames
            else:
                logger.warning("Audio capture reached max duration limit (%d s). Stopping.", self.max_seconds)
                self._is_capturing = False

    def start(self) -> None:
        """Start streaming microphone input into buffer."""
        with self._lock:
            self._buffer = io.BytesIO()
            self._frames_recorded = 0
            self._is_capturing = True

        if self._stream is None or not self._stream.active:
            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="int16",
                    callback=self._audio_callback,
                    blocksize=int(self.sample_rate * 0.03),  # 30ms chunks
                )
                self._stream.start()
                logger.debug("Audio stream started (16kHz mono).")
            except Exception as e:
                self._is_capturing = False
                logger.error("Failed to open microphone input stream: %s", e)
                raise

    def stop(self) -> bytes:
        """Stop capturing and return recorded raw 16-bit PCM bytes."""
        with self._lock:
            self._is_capturing = False
            raw_pcm = self._buffer.getvalue()
            self._buffer = io.BytesIO()
            self._frames_recorded = 0

        logger.debug("Audio capture stopped. Recorded %d bytes of PCM data.", len(raw_pcm))
        return raw_pcm

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
