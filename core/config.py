"""
Central Configuration Module for Laya Desktop Agent.
Handles loading, validation, and profile switching (low_resource, balanced, performance).
"""

from pathlib import Path
from typing import Literal, Optional
import os
import yaml
from pydantic import BaseModel, Field


class STTConfig(BaseModel):
    model_size: str = Field(default="small.en", description="faster-whisper model: tiny.en, base.en, small.en, large-v3")
    device: Literal["cuda", "cpu", "auto"] = Field(default="cuda", description="Preferred compute device")
    compute_type: Literal["float16", "int8_float16", "int8", "float32"] = Field(default="float16")
    idle_timeout_sec: int = Field(default=300, description="Seconds of inactivity before model is offloaded")
    language: str = Field(default="en")


class LayaConfig(BaseModel):
    model_id: str = Field(default="convai-innovations/laya-large-en", description="HuggingFace model ID or local path")
    device: Literal["cuda", "cpu", "auto"] = Field(default="cuda", description="Preferred device")
    compute_type: Literal["float16", "bfloat16", "float32"] = Field(default="float16")
    idle_timeout_sec: int = Field(default=300)
    confidence_threshold: float = Field(default=0.60, description="Minimum confidence for autonomous execution")


class TTSConfig(BaseModel):
    engine: Literal["piper"] = Field(default="piper")
    voice: str = Field(default="en_US-lessac-medium")
    device: Literal["cpu"] = Field(default="cpu", description="TTS is strictly pinned to CPU to preserve GPU VRAM")
    playback_volume: float = Field(default=1.0)
    enabled: bool = Field(default=True)


class VRAMConfig(BaseModel):
    max_agent_vram_mb: int = Field(default=2048, description="Maximum VRAM the agent is permitted to allocate")
    warning_threshold_pct: float = Field(default=75.0, description="Total GPU VRAM utilization triggering warning")
    critical_threshold_pct: float = Field(default=85.0, description="Total GPU VRAM utilization triggering emergency unload")
    poll_interval_sec: float = Field(default=2.0)
    auto_downgrade: bool = Field(default=True)


class RAMConfig(BaseModel):
    max_agent_ram_mb: int = Field(default=3072, description="Maximum agent RSS memory")
    warning_threshold_pct: float = Field(default=80.0)
    critical_threshold_pct: float = Field(default=90.0)


class CacheConfig(BaseModel):
    max_cache_mb: int = Field(default=500, description="Max total disk cache size")
    audio_ttl_sec: int = Field(default=30, description="Lifetime of temporary audio files")
    screenshot_ttl_sec: int = Field(default=60, description="Lifetime of temporary screenshots")
    cleanup_interval_sec: int = Field(default=60)


class InputConfig(BaseModel):
    ptt_key: str = Field(default="KEY_F12", description="Evdev Push-to-Talk keycode")
    emergency_stop_key: str = Field(default="KEY_F11", description="Evdev emergency abort keycode")
    sample_rate: int = Field(default=16000)
    channels: int = Field(default=1)
    vad_aggressiveness: int = Field(default=3, description="WebRTC VAD aggressiveness 0-3")


class SafetyConfig(BaseModel):
    require_confirmation_high_risk: bool = Field(default=True)
    require_confirmation_moderate: bool = Field(default=False)
    max_retries: int = Field(default=3)
    execution_timeout_sec: float = Field(default=15.0)


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")
    max_log_bytes: int = Field(default=5 * 1024 * 1024)  # 5 MB
    backup_count: int = Field(default=3)


class AgentConfig(BaseModel):
    profile: Literal["low_resource", "balanced", "performance"] = Field(default="balanced")
    stt: STTConfig = Field(default_factory=STTConfig)
    laya: LayaConfig = Field(default_factory=LayaConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    vram: VRAMConfig = Field(default_factory=VRAMConfig)
    ram: RAMConfig = Field(default_factory=RAMConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Path properties
    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def cache_dir(self) -> Path:
        return self.project_root / "cache"

    @property
    def audio_cache_dir(self) -> Path:
        d = self.cache_dir / "audio"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def screenshot_cache_dir(self) -> Path:
        d = self.cache_dir / "screenshots"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def model_cache_dir(self) -> Path:
        d = self.cache_dir / "models"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def logs_dir(self) -> Path:
        d = self.project_root / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def data_dir(self) -> Path:
        d = self.project_root / "data"
        d.mkdir(parents=True, exist_ok=True)
        return d


def load_config(config_path: Optional[Path] = None, profile: Optional[str] = None) -> AgentConfig:
    """Load configuration from YAML file and apply profile overrides."""
    root = Path(__file__).resolve().parent.parent
    
    if config_path is None:
        config_path = root / "configs" / "default.yaml"

    config_dict = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                config_dict = loaded

    # Handle profile override file if specified
    active_profile = profile or config_dict.get("profile", "balanced")
    profile_path = root / "configs" / f"{active_profile}.yaml"
    if profile_path.exists() and profile_path != config_path:
        with open(profile_path, "r", encoding="utf-8") as f:
            profile_dict = yaml.safe_load(f)
            if isinstance(profile_dict, dict):
                # Deep merge profile into config_dict
                for k, v in profile_dict.items():
                    if isinstance(v, dict) and k in config_dict and isinstance(config_dict[k], dict):
                        config_dict[k].update(v)
                    else:
                        config_dict[k] = v

    config_dict["profile"] = active_profile
    return AgentConfig.model_validate(config_dict)
