"""
Unit Tests for VRAM Guard and Resource Monitoring.
"""

from unittest.mock import MagicMock
import pytest
from core.config import VRAMConfig
from monitoring.resource_monitor import ResourceMonitor, ResourceSnapshot
from monitoring.vram_guard import VRAMGuard, VRAMStatus


def test_vram_guard_status_levels():
    config = VRAMConfig(
        max_agent_vram_mb=2048,
        warning_threshold_pct=75.0,
        critical_threshold_pct=85.0,
    )
    mock_monitor = MagicMock(spec=ResourceMonitor)

    # 1. Normal state (50% usage)
    mock_monitor.get_snapshot.return_value = ResourceSnapshot(
        gpu_available=True,
        vram_total_mb=6144.0,
        vram_used_mb=3072.0,
        vram_free_mb=3072.0,
        vram_used_pct=50.0,
        gpu_util_pct=10.0,
        gpu_temp_c=45.0,
        ram_total_mb=24000.0,
        ram_used_mb=6000.0,
        ram_free_mb=18000.0,
        ram_used_pct=25.0,
        swap_used_mb=0.0,
        cpu_percent=15.0,
        process_rss_mb=100.0,
        process_vms_mb=500.0,
    )

    guard = VRAMGuard(config, mock_monitor)
    assert guard.check_status() == VRAMStatus.NORMAL
    assert guard.can_allocate(1000.0) is True

    # 2. Warning state (78% usage)
    mock_monitor.get_snapshot.return_value.vram_used_pct = 78.0
    mock_monitor.get_snapshot.return_value.vram_used_mb = 4792.0
    assert guard.check_status() == VRAMStatus.WARNING
    # New allocation pushing beyond warning is rejected
    assert guard.can_allocate(200.0) is False

    # 3. Critical state (88% usage) triggers emergency cleanup callback
    callback_called = False
    def on_critical():
        nonlocal callback_called
        callback_called = True

    guard.register_unload_callback(on_critical)
    mock_monitor.get_snapshot.return_value.vram_used_pct = 88.0
    assert guard.check_status() == VRAMStatus.CRITICAL
    assert callback_called is True
