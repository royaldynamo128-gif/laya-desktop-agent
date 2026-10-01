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

import re

logger = logging.getLogger(__name__)

DEFAULT_COMMAND_PROMPT = (
    "Open Chrome. Open Dolphin. Open the text editor and write Hello World. "
    "Open Chrome and search for Linux. Open Settings. Switch to Telegram. "
    "Close this window."
)

HALLUCINATIONS = {
    "thank you.", "thank you", "thanks for watching.", "thanks for watching",
    "thank you very much.", "subscribe.", "please subscribe.", "bye.", "you",
    ".", "..", "...", "?", "!"
}


def is_hallucination(text: str) -> bool:
    clean = text.lower().strip()
    if clean in HALLUCINATIONS:
        return True
    if not re.sub(r"[^\w\s]", "", clean).strip():
        return True
    tokens = clean.split()
    if len(tokens) >= 3 and len(set(tokens)) == 1:
        return True
    return False


class TranscriptionResult(tuple):
    def __new__(cls, text: str, elapsed_ms: float, confidence: float = 1.0, language_prob: float = 1.0, raw_text: str = ""):
        return super().__new__(cls, (text, elapsed_ms))

    def __init__(self, text: str, elapsed_ms: float, confidence: float = 1.0, language_prob: float = 1.0, raw_text: str = ""):
        self.text = text
        self.raw_text = raw_text or text
        self.elapsed_ms = elapsed_ms
        self.confidence = confidence
        self.language_prob = language_prob


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

        start = time.perf_counter()
        try:
            if target_device == "cuda":
                # Ensure bundled nvidia CUDA 12 libraries are loaded globally
                import ctypes, sys
                from pathlib import Path
                nvidia_dir = Path(sys.executable).parent.parent / "lib/python3.12/site-packages/nvidia"
                if nvidia_dir.exists():
                    for rel in ["cublas/lib/libcublas.so.12", "cudnn/lib/libcudnn_ops.so.9"]:
                        lib_file = nvidia_dir / rel
                        if lib_file.exists():
                            try:
                                ctypes.CDLL(str(lib_file), mode=ctypes.RTLD_GLOBAL)
                            except Exception:
                                pass

            try:
                self._model = WhisperModel(
                    self.config.model_size,
                    device=target_device,
                    compute_type=target_compute,
                    download_root=self.download_root,
                    local_files_only=True,
                )
            except Exception:
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

    def transcribe(
        self,
        audio_data: Union[str, Path, bytes, np.ndarray],
        initial_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
        beam_size: Optional[int] = None,
    ) -> TranscriptionResult:
        """Transcribe audio with beam search, dynamic vocabulary biasing, and peak normalization.
        beam_size defaults to the configured value (1 = fast path for short commands)."""
        if beam_size is None:
            beam_size = getattr(self.config, "beam_size", 1)
        if not self.is_loaded:
            self.load()

        self._last_used_time = time.time()
        start = time.perf_counter()

        # Handle raw PCM bytes by converting to float32 numpy array
        if isinstance(audio_data, bytes):
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            input_audio = audio_array
        elif isinstance(audio_data, (str, Path)):
            import wave
            with wave.open(str(audio_data), "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                frames = wf.readframes(wf.getnframes())
                arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                if ch > 1:
                    arr = arr.reshape(-1, ch).mean(axis=1)
                if sr != 16000:
                    num_samples = int(len(arr) * 16000 / sr)
                    arr = np.interp(
                        np.linspace(0, len(arr), num_samples, endpoint=False),
                        np.arange(len(arr)),
                        arr,
                    ).astype(np.float32)
                input_audio = arr
        elif isinstance(audio_data, np.ndarray):
            if audio_data.dtype == np.int16:
                input_audio = audio_data.astype(np.float32) / 32768.0
            else:
                input_audio = audio_data
            input_audio = np.squeeze(input_audio)
        else:
            raise ValueError(f"Unsupported audio data type: {type(audio_data)}")

        if isinstance(input_audio, np.ndarray):
            input_audio = np.squeeze(input_audio)

        # Normalize non-silent audio to nominal level (~0.88 peak) for optimal Whisper attention
        if len(input_audio) > 0:
            peak = float(np.max(np.abs(input_audio)))
            if peak > 0.03:
                scale = min(0.88 / peak, 3.5)
                input_audio = np.clip(input_audio * scale, -1.0, 1.0)

        vad_params = dict(min_silence_duration_ms=250, speech_pad_ms=300)

        effective_prompt = initial_prompt or DEFAULT_COMMAND_PROMPT

        try:
            segments, info = self._model.transcribe(
                input_audio,
                language=self.config.language,
                beam_size=beam_size,
                best_of=beam_size,
                temperature=0.0,
                condition_on_previous_text=False,
                initial_prompt=effective_prompt,
                hotwords=hotwords,
                vad_filter=True,
                vad_parameters=vad_params,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            # Prevent hallucinating text on silence/ambient noise
            if getattr(info, "no_speech_prob", 0.0) > 0.60:
                logger.debug("High no-speech probability (%.2f). Rejecting hallucination.", info.no_speech_prob)
                return TranscriptionResult("", elapsed_ms, confidence=0.0, language_prob=getattr(info, "language_probability", 0.0), raw_text="")

            segments_list = list(segments)
            text = " ".join(seg.text for seg in segments_list).strip()

            if is_hallucination(text):
                logger.debug("Transcription '%s' identified as hallucination. Rejecting.", text)
                return TranscriptionResult("", elapsed_ms, confidence=0.0, language_prob=getattr(info, "language_probability", 0.0), raw_text=text)

            # Calculate STT confidence from average log probabilities
            stt_conf = 0.95
            if segments_list:
                logprobs = [getattr(s, "avg_logprob", -0.2) for s in segments_list]
                avg_lp = sum(logprobs) / len(logprobs)
                import math
                stt_conf = round(max(0.1, min(1.0, math.exp(avg_lp))), 3)

            logger.debug("Transcribed in %.1f ms: '%s' (conf: %.2f, prob: %.2f)", elapsed_ms, text, stt_conf, info.language_probability)
            return TranscriptionResult(text, elapsed_ms, confidence=stt_conf, language_prob=info.language_probability, raw_text=text)
        except Exception as e:
            if self._current_device == "cuda":
                logger.warning("CUDA transcription failed (%s). Falling back to CPU int8...", e)
                self.load(device="cpu", compute_type="int8")
                segments, info = self._model.transcribe(
                    input_audio,
                    language=self.config.language,
                    beam_size=beam_size,
                    best_of=beam_size,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    initial_prompt=effective_prompt,
                    hotwords=hotwords,
                    vad_filter=True,
                    vad_parameters=vad_params,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                if getattr(info, "no_speech_prob", 0.0) > 0.60:
                    return TranscriptionResult("", elapsed_ms, confidence=0.0, language_prob=getattr(info, "language_probability", 0.0), raw_text="")
                segments_list = list(segments)
                text = " ".join(seg.text for seg in segments_list).strip()
                if is_hallucination(text):
                    return TranscriptionResult("", elapsed_ms, confidence=0.0, language_prob=getattr(info, "language_probability", 0.0), raw_text=text)
                return TranscriptionResult(text, elapsed_ms, confidence=0.85, language_prob=info.language_probability, raw_text=text)
            else:
                raise

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
        if self.config.idle_timeout_sec <= 0:
            return False  # 0 or negative = never unload
        if self.is_loaded and (time.time() - self._last_used_time) > self.config.idle_timeout_sec:
            logger.info("Faster-Whisper exceeded idle timeout (%ds). Unloading.", self.config.idle_timeout_sec)
            self.unload()
            return True
        return False
