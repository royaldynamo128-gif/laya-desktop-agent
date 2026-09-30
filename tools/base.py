"""
Abstract Tool Base and Result Schema for Laya Desktop Agent.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel


class SafetyTier(str, Enum):
    SAFE = "SAFE"
    MODERATE = "MODERATE"
    HIGH_RISK = "HIGH_RISK"


@dataclass
class ToolResult:
    success: bool
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    observation_required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "error": self.error,
            "observation_required": self.observation_required,
        }


class Tool(ABC):
    name: str
    description: str
    safety_tier: SafetyTier = SafetyTier.SAFE

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool action with validated parameters."""
        pass
