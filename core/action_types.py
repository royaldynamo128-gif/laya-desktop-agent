"""
Neutral action type definitions shared across execution, planning, and
verification layers. Kept dependency-free to prevent circular imports.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class BoundAction:
    """A concrete tool invocation: tool name + validated arguments + target."""
    tool_name: str
    arguments: Dict[str, Any]
    target: str = ""
