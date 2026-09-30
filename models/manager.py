"""
Central Model Lifecycle Manager for Laya Desktop Agent.
Enforces LOAD -> WARM -> USE -> IDLE -> UNLOAD lifecycle, prevents model duplication,
and coordinates with VRAM Guard to execute automatic CPU fallback.
"""

from typing import Optional
import gc
import logging
import threading
import time
from core.config import AgentConfig
from monitoring.vram_guard import VRAMGuard, VRAMStatus
from models.whisper_engine import WhisperEngine
from models.laya_engine import LayaEngine
from models.piper_engine import PiperEngine

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, config: AgentConfig, vram_guard: VRAMGuard):
        self.config = config
        self.vram_guard = vram_guard

        # Initialize engines (unloaded initially)
        self.whisper = WhisperEngine(
            config=config.stt,
            download_root=config.model_cache_dir / "whisper",
        )
        self.laya = LayaEngine(
            config=config.laya,
        )
        self.piper = PiperEngine(
            config=config.tts,
            model_dir=config.model_cache_dir / "piper",
            audio_dir=config.audio_cache_dir,
        )

        # Register emergency unload callback with VRAM guard
        self.vram_guard.register_unload_callback(self.emergency_downgrade_to_cpu)

        # Idle sweep timer
        self._sweeper_active = False
        self._sweeper_thread: Optional[threading.Thread] = None

    def start_sweeper(self, interval_sec: float = 30.0) -> None:
        """Start background daemon thread to sweep idle models."""
        self._sweeper_active = True

        def _sweep_loop():
            while self._sweeper_active:
                time.sleep(interval_sec)
                try:
                    self.whisper.check_idle()
                    self.laya.check_idle()
                except Exception as e:
                    logger.debug("Error in model idle sweep: %s", e)

        self._sweeper_thread = threading.Thread(target=_sweep_loop, daemon=True, name="ModelIdleSweeper")
        self._sweeper_thread.start()

    def stop_sweeper(self) -> None:
        self._sweeper_active = False

    def warm_models(self) -> None:
        """Preload and warm up models for low-latency operation."""
        # Check VRAM headroom before loading on GPU
        status = self.vram_guard.check_status()
        
        # 1. Warm STT
        if status != VRAMStatus.CRITICAL and self.config.stt.device == "cuda":
            self.whisper.load(device="cuda")
        else:
            self.whisper.load(device="cpu")
        try:
            import numpy as np
            # Warm up whisper transcription on 0.5s of silence to allocate CUDA workspace
            self.whisper.transcribe(np.zeros(8000, dtype=np.float32))
        except Exception as e:
            logger.debug("STT warmup ping: %s", e)

        # 2. Warm Laya
        if status != VRAMStatus.CRITICAL and self.config.laya.device == "cuda":
            self.laya.load(device="cuda")
        else:
            self.laya.load(device="cpu")
        try:
            self.laya.decide("warmup ping")
        except Exception as e:
            logger.debug("Warmup inference ping: %s", e)

        # 3. Warm Piper on CPU
        if self.config.tts.enabled:
            self.piper.load()

        logger.info("All model engines initialized and warmed.")

    def emergency_downgrade_to_cpu(self) -> None:
        """Called automatically when GPU VRAM hits CRITICAL threshold."""
        logger.warning("VRAM Guard CRITICAL: Downgrading loaded models to CPU...")
        
        # Unload GPU models
        if self.whisper.is_loaded and self.whisper.current_device == "cuda":
            self.whisper.unload()
            # Reload on CPU in int8
            self.whisper.load(device="cpu", compute_type="int8")

        if self.laya.is_loaded and self.laya._current_device == "cuda":
            self.laya.unload()
            self.laya.load(device="cpu")

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("Emergency CPU fallback complete.")

    def shutdown(self) -> None:
        """Cleanly unload all models and terminate threads."""
        self.stop_sweeper()
        self.whisper.unload()
        self.laya.unload()
        self.piper.unload()
        gc.collect()
