"""
Health Checker for Laya Desktop Agent.
Validates dependencies, system interfaces (PipeWire, Wayland, kdotool, DBus, AT-SPI2, GPU),
and resource headroom.
"""

from typing import Dict, Any, List
import logging
import os
import shutil
import subprocess
from core.config import AgentConfig
from monitoring.resource_monitor import ResourceMonitor

logger = logging.getLogger(__name__)


class HealthChecker:
    def __init__(self, config: AgentConfig, monitor: ResourceMonitor):
        self.config = config
        self.monitor = monitor

    def run_all_checks(self) -> Dict[str, Any]:
        """Perform a complete non-blocking health check."""
        results: Dict[str, Any] = {}
        all_passed = True

        # 1. GPU & VRAM
        snapshot = self.monitor.get_snapshot()
        gpu_ok = snapshot.gpu_available and snapshot.vram_free_mb > 500
        results["gpu"] = {
            "status": "PASS" if gpu_ok else ("WARNING" if snapshot.gpu_available else "FAIL"),
            "available": snapshot.gpu_available,
            "vram_free_mb": snapshot.vram_free_mb,
            "vram_total_mb": snapshot.vram_total_mb,
        }
        if not gpu_ok and snapshot.gpu_available:
            all_passed = False

        # 2. Audio Subsystem (PipeWire / sounddevice)
        audio_ok = False
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            has_input = any(d.get("max_input_channels", 0) > 0 for d in devices)
            audio_ok = has_input
            results["audio"] = {
                "status": "PASS" if audio_ok else "FAIL",
                "devices_found": len(devices),
                "has_microphone": has_input,
            }
        except Exception as e:
            results["audio"] = {"status": "FAIL", "error": str(e)}
        if not audio_ok:
            all_passed = False

        # 3. Window Manager (kdotool for KDE Wayland)
        kdotool_path = shutil.which("kdotool") or "/usr/local/bin/kdotool"
        kdotool_ok = os.path.exists(kdotool_path) and os.access(kdotool_path, os.X_OK)
        results["kdotool"] = {
            "status": "PASS" if kdotool_ok else "FAIL",
            "path": kdotool_path,
        }
        if not kdotool_ok:
            all_passed = False

        # 4. Desktop DBus (qdbus6 / busctl)
        dbus_tool = shutil.which("qdbus6") or shutil.which("busctl")
        dbus_ok = dbus_tool is not None
        results["dbus"] = {
            "status": "PASS" if dbus_ok else "FAIL",
            "tool": dbus_tool,
        }
        if not dbus_ok:
            all_passed = False

        # 5. Accessibility (AT-SPI2 via gi.repository.Atspi)
        atspi_ok = False
        try:
            import gi
            gi.require_version("Atspi", "2.0")
            from gi.repository import Atspi
            desktop_count = Atspi.get_desktop_count()
            atspi_ok = desktop_count >= 0
            results["atspi"] = {"status": "PASS", "desktop_count": desktop_count}
        except Exception as e:
            results["atspi"] = {"status": "WARNING", "detail": f"AT-SPI2 fallback mode: {e}"}

        # 6. Disk space
        statvfs = os.statvfs(str(self.config.project_root))
        free_gb = (statvfs.f_bavail * statvfs.f_frsize) / (1024 ** 3)
        disk_ok = free_gb > 2.0
        results["disk"] = {
            "status": "PASS" if disk_ok else "FAIL",
            "free_gb": round(free_gb, 2),
        }
        if not disk_ok:
            all_passed = False

        # 7. Model Libraries
        libs_ok = True
        for lib in ["faster_whisper", "laya", "piper", "webrtcvad"]:
            try:
                __import__(lib)
            except Exception as e:
                libs_ok = False
                results[f"lib_{lib}"] = {"status": "FAIL", "error": str(e)}
        if libs_ok:
            results["libraries"] = {"status": "PASS"}
        else:
            all_passed = False

        results["overall_health"] = "PASS" if all_passed else "DEGRADED"
        return results
