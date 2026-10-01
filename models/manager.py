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
        self.vram_guard.register_restore_callback(self.restore_to_cuda)

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

        # Flush unused CUDA workspace allocations to keep base footprint minimal
        self.clean_vram()
        logger.info("All model engines initialized and warmed.")

    def clean_vram(self) -> dict:
        """Explicitly reclaim unallocated CUDA memory and collect garbage."""
        before_mb = 0.0
        after_mb = 0.0
        freed_mb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                before_mb = torch.cuda.memory_reserved(0) / (1024 * 1024)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                after_mb = torch.cuda.memory_reserved(0) / (1024 * 1024)
                freed_mb = max(0.0, before_mb - after_mb)
                logger.info("VRAM clean: reserved %.1f MB -> %.1f MB (freed %.1f MB)", before_mb, after_mb, freed_mb)
        except Exception as e:
            logger.warning("VRAM clean error: %s", e)
        return {"before_mb": before_mb, "after_mb": after_mb, "freed_mb": freed_mb}

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
                torch.cuda.ipc_collect()
        except ImportError:
            pass
        logger.info("Emergency CPU fallback complete.")

    def restore_to_cuda(self) -> None:
        """Called automatically when GPU VRAM normalizes to promote models back to CUDA."""
        if self.config.stt.device == "cuda" and self.whisper.current_device != "cuda":
            logger.info("Restoring Faster-Whisper to CUDA %s...", self.config.stt.compute_type)
            self.whisper.unload()
            self.whisper.load(device="cuda", compute_type=self.config.stt.compute_type)

        if self.config.laya.device == "cuda" and self.laya._current_device != "cuda":
            logger.info("Restoring Laya Router to CUDA...")
            self.laya.unload()
            self.laya.load(device="cuda")

        self.clean_vram()
        logger.info("Model restoration to CUDA complete.")

    def shutdown(self) -> None:
        """Cleanly unload all models and terminate threads."""
        self.stop_sweeper()
        self.whisper.unload()
        self.laya.unload()
        self.piper.unload()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass
