"""
Piper TTS Engine for Laya Desktop Agent.
High-speed, natural local text-to-speech pinned 100% to CPU via ONNX Runtime (0 MiB VRAM).
"""

from pathlib import Path
from typing import Optional, Union
import logging
import os
import time
import urllib.request
import wave
from piper.voice import PiperVoice
from core.config import TTSConfig

logger = logging.getLogger(__name__)

VOICE_URLS = {
    "en_US-lessac-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    },
    "en_US-ryan-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json",
    }
}


class PiperEngine:
    def __init__(self, config: TTSConfig, model_dir: Path, audio_dir: Path):
        self.config = config
        self.model_dir = model_dir
        self.audio_dir = audio_dir
        self._voice: Optional[PiperVoice] = None

    @property
    def is_loaded(self) -> bool:
        return self._voice is not None

    def _ensure_model_files(self) -> Path:
        """Download voice model files if not already cached."""
        voice_name = self.config.voice
        onnx_path = self.model_dir / f"{voice_name}.onnx"
        json_path = self.model_dir / f"{voice_name}.onnx.json"

        if onnx_path.exists() and json_path.exists():
            return onnx_path

        urls = VOICE_URLS.get(voice_name, VOICE_URLS["en_US-lessac-medium"])
        self.model_dir.mkdir(parents=True, exist_ok=True)

        if not json_path.exists():
            logger.info("Downloading Piper voice metadata: %s", json_path.name)
            urllib.request.urlretrieve(urls["json"], str(json_path))

        if not onnx_path.exists():
            logger.info("Downloading Piper voice ONNX weights: %s (~60 MB)", onnx_path.name)
            urllib.request.urlretrieve(urls["onnx"], str(onnx_path))

        return onnx_path

    def load(self) -> None:
        """Load Piper ONNX model into CPU memory."""
        if self._voice is not None:
            return

        onnx_path = self._ensure_model_files()
        start = time.perf_counter()
        logger.info("Loading Piper voice model [%s] on CPU...", onnx_path.name)
        self._voice = PiperVoice.load(str(onnx_path))
        logger.info("Piper TTS loaded in %.2f ms", (time.perf_counter() - start) * 1000.0)

    def synthesize_to_wav(self, text: str, output_wav_path: Optional[Path] = None) -> Path:
        """Synthesize text into a WAV file in <100ms."""
        if not self.config.enabled or not text.strip():
            return Path("")

        if not self.is_loaded:
            self.load()

        if output_wav_path is None:
            output_wav_path = self.audio_dir / f"tts_{int(time.time() * 1000)}.wav"

        start = time.perf_counter()
        with wave.open(str(output_wav_path), "wb") as wf:
            self._voice.synthesize_wav(text, wf)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.debug("Piper synthesized '%s' in %.1f ms -> %s", text, elapsed_ms, output_wav_path.name)
        return output_wav_path

    def unload(self) -> None:
        self._voice = None
