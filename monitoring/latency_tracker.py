"""
Latency Tracker for Laya Desktop Agent.
High-precision per-stage latency measurement, JSONL metrics recording, and statistics.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Generator, Optional
import json
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class CommandMetrics:
    command_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    transcription: str = ""
    intent: str = ""
    selected_tool: str = ""
    durations_ms: Dict[str, float] = field(default_factory=dict)
    total_ms: float = 0.0
    success: bool = True
    vram_used_mb: float = 0.0
    ram_used_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_id": self.command_id,
            "timestamp": self.timestamp,
            "transcription": self.transcription,
            "intent": self.intent,
            "selected_tool": self.selected_tool,
            "durations_ms": {k: round(v, 2) for k, v in self.durations_ms.items()},
            "total_ms": round(self.total_ms, 2),
            "success": self.success,
            "vram_used_mb": round(self.vram_used_mb, 1),
            "ram_used_mb": round(self.ram_used_mb, 1),
        }


class LatencyTracker:
    def __init__(self, metrics_file: Optional[Path] = None):
        self.metrics_file = metrics_file
        if self.metrics_file:
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def measure(self, stage_name: str, metrics: CommandMetrics) -> Generator[None, None, None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            metrics.durations_ms[stage_name] = elapsed_ms

    def record(self, metrics: CommandMetrics) -> None:
        """Calculate total latency and persist to rolling JSONL."""
        metrics.total_ms = sum(metrics.durations_ms.values())
        line = json.dumps(metrics.to_dict())
        logger.info(
            "Command [%s] completed in %.1f ms | Intent: %s | Tool: %s",
            metrics.command_id,
            metrics.total_ms,
            metrics.intent,
            metrics.selected_tool,
        )

        if self.metrics_file:
            try:
                with open(self.metrics_file, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception as e:
                logger.warning("Could not write metrics to %s: %s", self.metrics_file, e)
