"""
Pytest Fixtures and Mock Configurations for Laya Desktop Agent.
"""

import pytest
from pathlib import Path
from core.config import AgentConfig, load_config
from monitoring.resource_monitor import ResourceMonitor
from monitoring.vram_guard import VRAMGuard
from tools.registry import ToolRegistry
from safety.policy import SafetyPolicy
from safety.emergency_stop import EmergencyStop


@pytest.fixture
def config() -> AgentConfig:
    return load_config(profile="low_resource")


@pytest.fixture
def monitor() -> ResourceMonitor:
    return ResourceMonitor()


@pytest.fixture
def vram_guard(config: AgentConfig, monitor: ResourceMonitor) -> VRAMGuard:
    return VRAMGuard(config.vram, monitor)


@pytest.fixture
def emergency_stop() -> EmergencyStop:
    return EmergencyStop()


@pytest.fixture
def safety_policy(config: AgentConfig) -> SafetyPolicy:
    return SafetyPolicy(config.safety)


@pytest.fixture
def tool_registry() -> ToolRegistry:
    return ToolRegistry()
