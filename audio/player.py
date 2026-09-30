"""
Non-blocking Audio Player for Laya Desktop Agent.
Plays speech audio via SoundDevice / PipeWire, supporting instantaneous interrupt/stop.
"""

from pathlib import Path
from typing import Optional, Union
import logging
import threading
import wave
import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioPlayer:
    def __init__(self, volume: float = 1.0):
        self.volume = max(0.0, min(1.0, volume))
        self._lock = threading.Lock()
        self._current_stream = None

    def play_wav(self, wav_path: Union[str, Path], block: bool = False) -> None:
        """Play WAV audio file."""
        p = Path(wav_path)
        if not p.exists():
            logger.warning("WAV file does not exist: %s", p)
            return

        try:
            with wave.open(str(p), "rb") as wf:
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                data = np.frombuffer(frames, dtype=np.int16)
                if channels > 1:
                    data = data.reshape(-1, channels)

                self.play_array(data, sample_rate=sample_rate, block=block)
        except Exception as e:
            logger.error("Failed to play WAV file %s: %s", wav_path, e)

    def play_array(self, data: np.ndarray, sample_rate: int = 16000, block: bool = False) -> None:
        """Play numpy array audio data."""
        with self._lock:
            # Stop any ongoing playback
            self.stop()
            try:
                # Apply volume scaling
                scaled_data = data.astype(np.float32) / 32768.0 * self.volume
                sd.play(scaled_data, samplerate=sample_rate)
                if block:
                    sd.wait()
            except Exception as e:
                logger.error("Sounddevice playback error: %s", e)

    def stop(self) -> None:
        """Instantly stop any active playback (e.g. for emergency stop or interruption)."""
        try:
            sd.stop()
            logger.debug("Audio playback stopped.")
        except Exception as e:
            logger.debug("Error stopping playback: %s", e)
