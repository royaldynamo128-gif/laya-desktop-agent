"""
Pytest Fixtures and Mock Configurations for Laya Desktop Agent.

Hermetic test environment: all tests run against the simulated desktop in
tests/sim/ (fake kdotool window manager, fake .desktop application files,
fake wtype/xdg-open) — no dependency on the host's installed applications,
display server, or input devices.

Environment variables are set at conftest import time (before any laya
module imports read them).
"""

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Hermetic environment setup — MUST run before laya module imports
# ---------------------------------------------------------------------------
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
SIM_DIR = TESTS_DIR / "sim"
SIM_BIN = SIM_DIR / "bin"
SIM_APPS = SIM_DIR / "apps"
SIM_STATE = SIM_DIR / "state"

os.environ["KDTOOL_BIN"] = str(SIM_BIN / "kdotool")
os.environ["KDOTOOL_STATE"] = str(SIM_STATE / "wm_state.json")
os.environ["LAYA_DESKTOP_DIRS"] = str(SIM_APPS)
os.environ["WTYPE_BIN"] = str(SIM_BIN / "wtype")
os.environ["INPUT_LOG"] = str(SIM_STATE / "input_log.jsonl")
os.environ["BROWSER_LOG"] = str(SIM_STATE / "browser_log.jsonl")
os.environ["FAKEAPP_UI_LOG"] = str(SIM_STATE / "app_log.jsonl")
os.environ["FAKEAPP_LAUNCH_DELAY_MS"] = "200"
os.environ["FAKEBROWSER_OPEN_DELAY_MS"] = "100"
os.environ.setdefault("LAYA_OVERLAY_LD_PRELOAD", "")  # no layer-shell in tests

# Put sim binaries on PATH (fake xdg-open, gtk-launch, app wrappers)
os.environ["PATH"] = f"{SIM_BIN}:{os.environ.get('PATH', '')}"

# Project root on sys.path for package imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# In hermetic simulated environment, ensure input synthesis routes to simulated wtype
try:
    import tools.input_control
    tools.input_control.HAS_ATSPI = False
except Exception:
    pass


def _reset_sim_state() -> None:
    """Reset the simulated window manager and all logs."""
    SIM_STATE.mkdir(parents=True, exist_ok=True)
    state = {"windows": {}, "active": None, "desktop": 1, "log": []}
    with open(SIM_STATE / "wm_state.json", "w") as f:
        json.dump(state, f)
    for log in ("input_log.jsonl", "browser_log.jsonl", "app_log.jsonl"):
        p = SIM_STATE / log
        if p.exists():
            p.unlink()
    # Screen-state + action logs for the fake AT-SPI
    for p in (
        SIM_STATE / "fake_screen.json",
        SIM_STATE / "atspi_actions.jsonl",
        SIM_STATE / "wm_state.json.failcount",
    ):
        if p.exists():
            p.unlink()


def _clear_app_caches() -> None:
    """Remove machine-specific app catalog caches so tests always index the
    simulated applications directory fresh."""
    for cache in (PROJECT_ROOT / "data" / "app_catalog.json", PROJECT_ROOT / "data" / "app_registry.json"):
        if cache.exists():
            cache.unlink()


os.environ.setdefault("ATSPI_SCREEN_STATE", str(SIM_STATE / "fake_screen.json"))
os.environ.setdefault("ATSPI_ACTION_LOG", str(SIM_STATE / "atspi_actions.jsonl"))


def _kill_fake_apps() -> None:
    """Terminate leftover fakeapp processes between tests."""
    try:
        import subprocess

        res = subprocess.run(
            ["pgrep", "-f", "sim/bin/fakeapp"], capture_output=True, text=True, timeout=5
        )
        pids = [int(p) for p in res.stdout.split() if p.strip().isdigit()]
        for pid in pids:
            try:
                os.kill(pid, 15)
            except Exception:
                pass
    except Exception:
        pass


# Reset before every test module (and any stray fakeapps)
_reset_sim_state()
_clear_app_caches()


@pytest.fixture(autouse=True)
def _clean_sim_desktop():
    """Each test starts with a clean simulated desktop."""
    _kill_fake_apps()
    _reset_sim_state()
    # Clear the app catalog cache instance between tests (module-level parser
    # is fine to reuse; catalogs pick up the env var at construction)
    yield
    _kill_fake_apps()


# ---------------------------------------------------------------------------
# Laya fixtures
# ---------------------------------------------------------------------------

from core.config import AgentConfig, load_config  # noqa: E402
from monitoring.resource_monitor import ResourceMonitor  # noqa: E402
from monitoring.vram_guard import VRAMGuard  # noqa: E402
from tools.registry import ToolRegistry  # noqa: E402
from safety.policy import SafetyPolicy  # noqa: E402
from safety.emergency_stop import EmergencyStop  # noqa: E402


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


@pytest.fixture
def sim_catalog():
    """AppCatalog bound to the simulated applications directory (no cache)."""
    from integrations.app_catalog import AppCatalog

    return AppCatalog(desktop_dirs=[SIM_APPS])
