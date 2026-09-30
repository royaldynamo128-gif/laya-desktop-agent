"""
VRAM Guard for Laya Desktop Agent.
Enforces strict 6 GB GPU limits, provides 3-tier memory status (NORMAL, WARNING, CRITICAL),
and orchestrates emergency cache flushes and CPU downgrades.
"""

from enum import Enum
from typing import Callable, List, Optional
import gc
import logging
from core.config import VRAMConfig
from monitoring.resource_monitor import ResourceMonitor, ResourceSnapshot

logger = logging.getLogger(__name__)


class VRAMStatus(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class VRAMGuard:
    def __init__(self, config: VRAMConfig, monitor: Optional[ResourceMonitor] = None):
        self.config = config
        self.monitor = monitor or ResourceMonitor()
        self._unload_callbacks: List[Callable[[], None]] = []
        self._last_status = VRAMStatus.NORMAL

    def register_unload_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback to release GPU models during CRITICAL memory pressure."""
        if callback not in self._unload_callbacks:
            self._unload_callbacks.append(callback)

    def check_status(self) -> VRAMStatus:
        """Inspect current VRAM utilization and trigger defensive actions if thresholds are exceeded."""
        snapshot = self.monitor.get_snapshot()
        if not snapshot.gpu_available:
            return VRAMStatus.NORMAL

        used_pct = snapshot.vram_used_pct
        status = VRAMStatus.NORMAL

        if used_pct >= self.config.critical_threshold_pct:
            status = VRAMStatus.CRITICAL
        elif used_pct >= self.config.warning_threshold_pct:
            status = VRAMStatus.WARNING

        if status != self._last_status:
            logger.warning(
                "VRAM State Changed: %s -> %s (Used: %.1f MB / %.1f MB, %.1f%%)",
                self._last_status.value,
                status.value,
                snapshot.vram_used_mb,
                snapshot.vram_total_mb,
                used_pct,
            )
            self._last_status = status

        if status == VRAMStatus.CRITICAL:
            self.emergency_cleanup()

        return status

    def can_allocate(self, required_mb: float) -> bool:
        """Check whether allocating required_mb of VRAM will safely stay below thresholds."""
        snapshot = self.monitor.get_snapshot()
        if not snapshot.gpu_available:
            return False

        projected_used_mb = snapshot.vram_used_mb + required_mb
        projected_pct = (projected_used_mb / snapshot.vram_total_mb) * 100.0

        if projected_pct > self.config.warning_threshold_pct:
            logger.warning(
                "Refusing GPU allocation of %.1f MB: projected usage %.1f%% exceeds threshold %.1f%%",
                required_mb,
                projected_pct,
                self.config.warning_threshold_pct,
            )
            return False

        return True

    def emergency_cleanup(self) -> None:
        """Execute emergency GPU memory reclamation."""
        logger.error("Triggering EMERGENCY VRAM cleanup!")

        # 1. Trigger registered model offloaders (Whisper, etc.)
        for cb in self._unload_callbacks:
            try:
                cb()
            except Exception as e:
                logger.error("Error in unload callback: %s", e)

        # 2. Collect Python garbage
        gc.collect()

        # 3. Clear PyTorch CUDA cache if torch is imported
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                logger.info("PyTorch CUDA cache flushed successfully.")
        except ImportError:
            pass

        # Verify new memory level
        snapshot = self.monitor.get_snapshot()
        logger.info(
            "Post-cleanup VRAM: %.1f MB used / %.1f MB free (%.1f%%)",
            snapshot.vram_used_mb,
            snapshot.vram_free_mb,
            snapshot.vram_used_pct,
        )
