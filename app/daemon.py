"""
Main Background Daemon for Laya Desktop Agent.
Enforces single-instance protection, wires the entire architecture, and runs event loop.
"""

from pathlib import Path
from typing import Optional
import fcntl
import logging
import os
import signal
import sys
import time

from core.config import AgentConfig, load_config
from core.state import StateManager, AgentState
from core.events import EventBus
from core.coordinator import PipelineCoordinator
from monitoring.resource_monitor import ResourceMonitor
from monitoring.vram_guard import VRAMGuard
from monitoring.cache_reaper import CacheReaper
from monitoring.latency_tracker import LatencyTracker
from monitoring.health_checker import HealthChecker
from audio.capture import AudioCapture
from audio.vad import VoiceActivityDetector
from audio.player import AudioPlayer
from models.manager import ModelManager
from integrations.kdotool_client import KdotoolClient
from integrations.flatpak_helper import AppLauncher
from integrations.atspi_client import AtspiClient
from integrations.dbus_client import DBusClient
from integrations.playwright_client import BrowserClient
from tools.registry import ToolRegistry
from tools.app_control import OpenApplicationTool, CloseApplicationTool, SwitchWindowTool
from tools.window_control import MinimizeWindowTool, MaximizeWindowTool, SwitchDesktopTool
from tools.browser_control import BrowserSearchTool, BrowserNavigateTool
from tools.input_control import ScrollTool, TypeTextTool, PressKeyTool
from tools.screen_control import ReadScreenTool, TakeScreenshotTool
from tools.system_control import SetVolumeTool, MuteVolumeTool, MediaControlTool
from execution.parameter_binder import ParameterBinder
from execution.executor import ActionExecutor
from safety.policy import SafetyPolicy
from safety.emergency_stop import EmergencyStop
from verification.observer import DesktopObserver
from verification.verifier import ActionVerifier
from verification.recovery import RecoveryManager
from app.ptt_listener import PTTListener

logger = logging.getLogger("laya_agent")


class LayaDaemon:
    def __init__(self, config: AgentConfig, diagnostic_mode: bool = False):
        self.config = config
        self.diagnostic_mode = diagnostic_mode
        self._lock_file = None
        self._running = False

        # 1. State & Monitoring
        self.state_mgr = StateManager()
        self.monitor = ResourceMonitor()
        self.vram_guard = VRAMGuard(config.vram, self.monitor)
        self.reaper = CacheReaper(config)
        self.tracker = LatencyTracker(config.logs_dir / "metrics.jsonl")
        self.emergency_stop = EmergencyStop()
        self.safety_policy = SafetyPolicy(config.safety)

        # 2. Audio & Models
        self.audio_capture = AudioCapture(
            sample_rate=config.input.sample_rate,
            channels=config.input.channels,
        )
        self.vad = VoiceActivityDetector(
            sample_rate=config.input.sample_rate,
            aggressiveness=config.input.vad_aggressiveness,
        )
        self.player = AudioPlayer(volume=config.tts.playback_volume)
        self.model_mgr = ModelManager(config, self.vram_guard)

        # Register emergency stop callback with audio player
        self.emergency_stop.register_callback(self.player.stop)

        # 3. Integrations
        self.kdotool = KdotoolClient()
        self.launcher = AppLauncher(cache_file=config.data_dir / "app_registry.json")
        self.atspi = AtspiClient()
        self.dbus = DBusClient()
        self.browser = BrowserClient()

        # 4. Tools Registry
        self.registry = ToolRegistry()
        self._register_all_tools()

        # 5. Execution & Verification
        self.binder = ParameterBinder()
        self.executor = ActionExecutor(self.registry, self.safety_policy, self.emergency_stop, config.safety)
        self.observer = DesktopObserver(self.kdotool, self.atspi)
        self.verifier = ActionVerifier()
        self.recovery_mgr = RecoveryManager(max_retries=config.safety.max_retries)

        # 6. Pipeline Coordinator
        self.coordinator = PipelineCoordinator(
            config=config,
            state_mgr=self.state_mgr,
            model_mgr=self.model_mgr,
            vad=self.vad,
            player=self.player,
            binder=self.binder,
            executor=self.executor,
            observer=self.observer,
            verifier=self.verifier,
            recovery_mgr=self.recovery_mgr,
            tracker=self.tracker,
            reaper=self.reaper,
            monitor=self.monitor,
            diagnostic_mode=diagnostic_mode,
        )

        # 7. Global Key Listener
        self.ptt_listener = PTTListener(
            ptt_key_name=config.input.ptt_key,
            estop_key_name=config.input.emergency_stop_key,
            on_press=self._on_ptt_press,
            on_release=self._on_ptt_release,
            on_estop=self._on_estop_press,
        )

    def _register_all_tools(self) -> None:
        self.registry.register(OpenApplicationTool(self.launcher, self.kdotool))
        self.registry.register(CloseApplicationTool(self.kdotool))
        self.registry.register(SwitchWindowTool(self.kdotool))
        self.registry.register(MinimizeWindowTool(self.kdotool))
        self.registry.register(MaximizeWindowTool(self.kdotool))
        self.registry.register(SwitchDesktopTool(self.kdotool))
        self.registry.register(BrowserSearchTool(self.browser))
        self.registry.register(BrowserNavigateTool(self.browser))
        self.registry.register(ScrollTool())
        self.registry.register(TypeTextTool())
        self.registry.register(PressKeyTool())
        self.registry.register(ReadScreenTool(self.atspi, self.kdotool))
        self.registry.register(TakeScreenshotTool(self.config.screenshot_cache_dir))
        self.registry.register(SetVolumeTool())
        self.registry.register(MuteVolumeTool())
        self.registry.register(MediaControlTool(self.dbus))

    def _acquire_single_instance_lock(self) -> bool:
        """Enforce single-instance protection via flock."""
        runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
        lock_path = runtime_dir / "laya_voice_agent.lock"

        try:
            self._lock_file = open(lock_path, "w")
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock_file.write(str(os.getpid()))
            self._lock_file.flush()
            logger.info("Single instance lock acquired at: %s (PID: %d)", lock_path, os.getpid())
            return True
        except (IOError, OSError):
            logger.error("Another instance of Laya Desktop Agent is already running! Exiting.")
            return False

    def _on_ptt_press(self) -> None:
        """Called when user presses and holds Push-to-Talk key."""
        if self.emergency_stop.is_stopped():
            logger.warning("Ignoring PTT: Agent is in Emergency Stop state.")
            return

        logger.info("Push-to-Talk active: recording...")
        self.state_mgr.set_state(AgentState.LISTENING, reason="PTT key down")
        self.audio_capture.start()

    def _on_ptt_release(self) -> None:
        """Called when user releases Push-to-Talk key."""
        if not self.audio_capture.is_capturing:
            return

        logger.info("Push-to-Talk released: processing speech...")
        raw_pcm = self.audio_capture.stop()
        if raw_pcm:
            # Process in thread to avoid blocking evdev listener
            import threading
            threading.Thread(
                target=self.coordinator.process_audio,
                args=(raw_pcm,),
                daemon=True,
                name="AudioProcessor",
            ).start()

    def _on_estop_press(self) -> None:
        """Called when user hits Emergency Stop key."""
        self.emergency_stop.trigger(reason="Global hotkey F11 pressed")
        self.state_mgr.set_state(AgentState.PAUSED, reason="Emergency Stop")
        self.audio_capture.stop()
        self.dbus.show_notification("Laya Agent", "EMERGENCY STOP ACTIVATED", timeout_ms=4000)

    def start(self) -> None:
        """Start daemon and enter main loop."""
        if not self._acquire_single_instance_lock():
            sys.exit(1)

        logger.info("Starting Laya Voice Agent Daemon (Diagnostic Mode: %s)...", self.diagnostic_mode)

        # Setup OS signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # Pre-warm models
        self.model_mgr.warm_models()
        self.model_mgr.start_sweeper()

        # Start PTT listener
        self.ptt_listener.start()

        self._running = True
        self.state_mgr.set_state(AgentState.IDLE, reason="Daemon started")
        self.dbus.show_notification("Laya Agent", "Voice Desktop Agent Online & Ready (F12 = Talk, F11 = Stop)")

        logger.info("Laya Desktop Agent is running. Hold F12 to talk, F11 for Emergency Stop.")

        try:
            while self._running:
                time.sleep(1.0)
                # Periodic VRAM & health check
                self.vram_guard.check_status()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _handle_signal(self, signum, frame):
        logger.info("Received termination signal (%d). Shutting down cleanly...", signum)
        self._running = False

    def stop(self) -> None:
        """Clean shutdown of all subsystems."""
        logger.info("Stopping Laya Desktop Agent...")
        self._running = False
        self.ptt_listener.stop()
        self.audio_capture.close()
        self.model_mgr.shutdown()
        self.monitor.shutdown()
        self.executor.shutdown()

        if self._lock_file:
            try:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                self._lock_file.close()
            except Exception:
                pass

        logger.info("Laya Desktop Agent stopped.")
