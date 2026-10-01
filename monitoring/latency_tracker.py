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
    raw_transcription: str = ""
    target: str = ""
    selected_tool: str = ""
    audio_quality_conf: float = 1.0
    stt_confidence: float = 1.0
    entity_resolution_conf: float = 1.0
    intent_confidence: float = 1.0
    durations_ms: Dict[str, float] = field(default_factory=dict)
    total_ms: float = 0.0
    success: bool = True
    vram_used_mb: float = 0.0
    ram_used_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        durations = getattr(self, "durations_ms", {}) or {}
        return {
            "command_id": getattr(self, "command_id", ""),
            "timestamp": getattr(self, "timestamp", ""),
            "raw_transcription": getattr(self, "raw_transcription", ""),
            "transcription": getattr(self, "transcription", ""),
            "target": getattr(self, "target", ""),
            "intent": getattr(self, "intent", ""),
            "selected_tool": getattr(self, "selected_tool", ""),
            "audio_quality_conf": round(getattr(self, "audio_quality_conf", 1.0), 2),
            "stt_confidence": round(getattr(self, "stt_confidence", 1.0), 2),
            "entity_resolution_conf": round(getattr(self, "entity_resolution_conf", 1.0), 2),
            "intent_confidence": round(getattr(self, "intent_confidence", 1.0), 2),
            "durations_ms": {k: round(v, 2) for k, v in durations.items()},
            "total_ms": round(getattr(self, "total_ms", 0.0), 2),
            "success": getattr(self, "success", True),
            "vram_used_mb": round(getattr(self, "vram_used_mb", 0.0), 1),
            "ram_used_mb": round(getattr(self, "ram_used_mb", 0.0), 1),
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
