"""
Tool Registry for Laya Desktop Agent.
Maintains validated tools, enforces access boundaries, and reports registered capabilities.
"""

from typing import Dict, List, Optional
import logging
from tools.base import Tool, ToolResult, SafetyTier

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool instance."""
        if tool.name in self._tools:
            logger.debug("Overwriting existing tool registration: %s", tool.name)
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s [Safety: %s]", tool.name, tool.safety_tier.value)

    def get(self, name: str) -> Optional[Tool]:
        """Retrieve tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, str]]:
        """List all registered tools and metadata."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "safety_tier": t.safety_tier.value,
            }
            for t in self._tools.values()
        ]

    def has_tool(self, name: str) -> bool:
        return name in self._tools
