"""
Unit Tests for Safety Policy and Permission Gating.
"""

import pytest
from core.config import SafetyConfig
from tools.base import Tool, ToolResult, SafetyTier
from safety.policy import SafetyPolicy


class SafeDummyTool(Tool):
    name = "safe_dummy"
    description = "Safe tool"
    safety_tier = SafetyTier.SAFE

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, message="OK")


class ModerateDummyTool(Tool):
    name = "moderate_dummy"
    description = "Moderate tool"
    safety_tier = SafetyTier.MODERATE

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, message="OK")


class HighRiskDummyTool(Tool):
    name = "high_risk_dummy"
    description = "Destructive tool"
    safety_tier = SafetyTier.HIGH_RISK

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, message="OK")


def test_safety_policy_tiers():
    config = SafetyConfig(require_confirmation_high_risk=True, require_confirmation_moderate=True)
    policy = SafetyPolicy(config)

    safe_tool = SafeDummyTool()
    mod_tool = ModerateDummyTool()
    high_tool = HighRiskDummyTool()

    # Safe tool is always permitted without confirmation
    permitted, _ = policy.check_permission(safe_tool, is_confirmed=False)
    assert permitted is True

    # Moderate tool is blocked without confirmation when required
    permitted, _ = policy.check_permission(mod_tool, is_confirmed=False)
    assert permitted is False
    # Moderate tool is permitted when confirmed
    permitted, _ = policy.check_permission(mod_tool, is_confirmed=True)
    assert permitted is True

    # High risk tool is strictly blocked without confirmation
    permitted, _ = policy.check_permission(high_tool, is_confirmed=False)
    assert permitted is False

    # High risk tool is permitted when confirmed
    permitted, _ = policy.check_permission(high_tool, is_confirmed=True)
    assert permitted is True
