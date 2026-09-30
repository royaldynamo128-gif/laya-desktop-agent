"""
Safety Policy and Permission Gating for Laya Desktop Agent.
Enforces 3-tier safety levels (SAFE, MODERATE, HIGH_RISK) and protects against accidental damage.
"""

from typing import Tuple
import logging
from core.config import SafetyConfig
from tools.base import Tool, SafetyTier

logger = logging.getLogger(__name__)


class SafetyPolicy:
    def __init__(self, config: SafetyConfig):
        self.config = config

    def check_permission(self, tool: Tool, is_confirmed: bool = False) -> Tuple[bool, str]:
        """Verify whether an action is permitted under current safety policy."""
        tier = tool.safety_tier

        # 1. High Risk Actions: Require explicit user confirmation every time
        if tier == SafetyTier.HIGH_RISK:
            if not is_confirmed:
                msg = f"Action '{tool.name}' is HIGH RISK and requires explicit confirmation."
                logger.warning(msg)
                return False, msg
            return True, "High risk action confirmed by user."

        # 2. Moderate Actions: Require confirmation only if configured
        if tier == SafetyTier.MODERATE:
            if self.config.require_confirmation_moderate and not is_confirmed:
                msg = f"Action '{tool.name}' is MODERATE RISK and requires confirmation."
                logger.info(msg)
                return False, msg
            return True, "Moderate risk action permitted."

        # 3. Safe Actions: Always permitted automatically
        return True, "Safe action permitted automatically."
