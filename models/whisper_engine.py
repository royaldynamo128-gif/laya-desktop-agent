"""
Faster-Whisper STT Engine for Laya Desktop Agent.
Supports dynamic device switching (CUDA float16 <-> CPU int8) and idle offloading.
"""

from pathlib import Path
from typing import Optional, Union, Tuple
import gc
import io
import logging
import time
import numpy as np
from faster_whisper import WhisperModel
from core.config import STTConfig

logger = logging.getLogger(__name__)


class WhisperEngine:
    def __init__(self, config: STTConfig, download_root: Optional[Path] = None):
        self.config = config
        self.download_root = str(download_root) if download_root else None
        self._model: Optional[WhisperModel] = None
        self._current_device: str = config.device
        self._current_compute_type: str = config.compute_type
        self._last_used_time: float = 0.0

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def current_device(self) -> str:
        return self._current_device

    def load(self, device: Optional[str] = None, compute_type: Optional[str] = None) -> None:
        """Load or reload model on requested device."""
        target_device = device or self.config.device
        target_compute = compute_type or self.config.compute_type

        # Auto-detect cuda availability
        if target_device == "auto":
            try:
                import torch
                target_device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                target_device = "cpu"

        if target_device == "cpu":
            target_compute = "int8"

        # If already loaded on same device and compute_type, just return
        if self._model is not None and self._current_device == target_device and self._current_compute_type == target_compute:
            return

        logger.info(
            "Loading Faster-Whisper [%s] on %s (%s)...",
            self.config.model_size,
            target_device,
            target_compute,
        )

        # Unload previous instance if needed
        self.unload()

        start = time.perf_counter()
        try:
            self._model = WhisperModel(
                self.config.model_size,
                device=target_device,
                compute_type=target_compute,
                download_root=self.download_root,
            )
            self._current_device = target_device
            self._current_compute_type = target_compute
            self._last_used_time = time.time()
            elapsed = time.perf_counter() - start
            logger.info("Faster-Whisper loaded in %.2f s on %s", elapsed, target_device)
        except Exception as e:
            if target_device == "cuda":
                logger.warning("CUDA load failed for Whisper (%s). Falling back to CPU...", e)
                self.load(device="cpu", compute_type="int8")
            else:
                logger.error("Failed to load Whisper on CPU: %s", e)
                raise

    def transcribe(self, audio_data: Union[str, Path, bytes, np.ndarray]) -> Tuple[str, float]:
        """Transcribe audio (file path, raw PCM bytes, or float32/int16 array) to text."""
        if not self.is_loaded:
            self.load()

        self._last_used_time = time.time()
        start = time.perf_counter()

        # Handle raw PCM bytes by converting to float32 numpy array
        if isinstance(audio_data, bytes):
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            input_audio = audio_array
        elif isinstance(audio_data, (str, Path)):
            input_audio = str(audio_data)
        elif isinstance(audio_data, np.ndarray):
            if audio_data.dtype == np.int16:
                input_audio = audio_data.astype(np.float32) / 32768.0
            else:
                input_audio = audio_data
        else:
            raise ValueError(f"Unsupported audio data type: {type(audio_data)}")

        segments, info = self._model.transcribe(
            input_audio,
            language=self.config.language,
            beam_size=1,  # fast single-beam for low latency
            vad_filter=False,  # we already run WebRTC VAD
        )

        text = " ".join(seg.text for seg in segments).strip()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.debug("Transcribed in %.1f ms: '%s' (prob: %.2f)", elapsed_ms, text, info.language_probability)
        return text, elapsed_ms

    def unload(self) -> None:
        """Explicitly release model from memory."""
        if self._model is not None:
            logger.info("Unloading Faster-Whisper model from %s...", self._current_device)
            self._model = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def check_idle(self) -> bool:
        """Check if model has exceeded idle timeout and unload if so."""
        if self.is_loaded and (time.time() - self._last_used_time) > self.config.idle_timeout_sec:
            logger.info("Faster-Whisper exceeded idle timeout (%ds). Unloading.", self.config.idle_timeout_sec)
            self.unload()
            return True
        return False
