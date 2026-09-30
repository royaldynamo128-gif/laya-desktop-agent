"""
Resource Monitor for Laya Desktop Agent.
Polls GPU VRAM (via NVML) and host RAM/CPU (via psutil) with zero overhead.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging
import os
import psutil

# Safe NVML import with graceful fallback
import warnings
try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import pynvml
    HAS_NVML = True
except ImportError:
    try:
        import nvidia_smi as pynvml
        HAS_NVML = True
    except ImportError:
        HAS_NVML = False

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    # GPU / VRAM
    gpu_available: bool
    vram_total_mb: float
    vram_used_mb: float
    vram_free_mb: float
    vram_used_pct: float
    gpu_util_pct: float
    gpu_temp_c: float
    # System RAM
    ram_total_mb: float
    ram_used_mb: float
    ram_free_mb: float
    ram_used_pct: float
    swap_used_mb: float
    cpu_percent: float
    # Agent Process
    process_rss_mb: float
    process_vms_mb: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpu_available": self.gpu_available,
            "vram_total_mb": round(self.vram_total_mb, 1),
            "vram_used_mb": round(self.vram_used_mb, 1),
            "vram_free_mb": round(self.vram_free_mb, 1),
            "vram_used_pct": round(self.vram_used_pct, 1),
            "gpu_util_pct": round(self.gpu_util_pct, 1),
            "gpu_temp_c": round(self.gpu_temp_c, 1),
            "ram_total_mb": round(self.ram_total_mb, 1),
            "ram_used_mb": round(self.ram_used_mb, 1),
            "ram_free_mb": round(self.ram_free_mb, 1),
            "ram_used_pct": round(self.ram_used_pct, 1),
            "swap_used_mb": round(self.swap_used_mb, 1),
            "cpu_percent": round(self.cpu_percent, 1),
            "process_rss_mb": round(self.process_rss_mb, 1),
        }


class ResourceMonitor:
    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._nvml_initialized = False
        self._nvml_handle = None
        self._process = psutil.Process(os.getpid())
        self._init_nvml()

    def _init_nvml(self) -> None:
        if not HAS_NVML:
            logger.warning("NVML package not available. GPU monitoring disabled.")
            return

        try:
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._nvml_initialized = True
            name = pynvml.nvmlDeviceGetName(self._nvml_handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            logger.info("NVML initialized for GPU %d: %s", self.device_index, name)
        except Exception as e:
            logger.warning("Could not initialize NVML for GPU %d: %s", self.device_index, e)
            self._nvml_initialized = False

    def get_snapshot(self) -> ResourceSnapshot:
        """Capture a comprehensive instantaneous resource snapshot."""
        # System RAM & CPU
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        cpu_pct = psutil.cpu_percent(interval=None)
        
        proc_mem = self._process.memory_info()
        proc_rss_mb = proc_mem.rss / (1024 * 1024)
        proc_vms_mb = proc_mem.vms / (1024 * 1024)

        # GPU metrics
        gpu_avail = False
        vram_total_mb = 0.0
        vram_used_mb = 0.0
        vram_free_mb = 0.0
        vram_used_pct = 0.0
        gpu_util_pct = 0.0
        gpu_temp_c = 0.0

        if self._nvml_initialized and self._nvml_handle:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                vram_total_mb = mem_info.total / (1024 * 1024)
                vram_used_mb = mem_info.used / (1024 * 1024)
                vram_free_mb = mem_info.free / (1024 * 1024)
                vram_used_pct = (vram_used_mb / vram_total_mb * 100.0) if vram_total_mb > 0 else 0.0

                util_info = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                gpu_util_pct = float(util_info.gpu)

                gpu_temp_c = float(pynvml.nvmlDeviceGetTemperature(
                    self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU
                ))
                gpu_avail = True
            except Exception as e:
                logger.debug("Failed to read NVML metrics: %s", e)

        return ResourceSnapshot(
            gpu_available=gpu_avail,
            vram_total_mb=vram_total_mb,
            vram_used_mb=vram_used_mb,
            vram_free_mb=vram_free_mb,
            vram_used_pct=vram_used_pct,
            gpu_util_pct=gpu_util_pct,
            gpu_temp_c=gpu_temp_c,
            ram_total_mb=mem.total / (1024 * 1024),
            ram_used_mb=mem.used / (1024 * 1024),
            ram_free_mb=mem.available / (1024 * 1024),
            ram_used_pct=mem.percent,
            swap_used_mb=swap.used / (1024 * 1024),
            cpu_percent=cpu_pct,
            process_rss_mb=proc_rss_mb,
            process_vms_mb=proc_vms_mb,
        )

    def shutdown(self) -> None:
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False
