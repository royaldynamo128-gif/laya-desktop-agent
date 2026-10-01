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
import subprocess
import sys
import threading
import time

from core.config import AgentConfig, load_config
from core.state import StateManager, AgentState
from core.events import EventBus
from core.coordinator import PipelineCoordinator
from core.status_ipc import StatusBroadcaster
from core.lifecycle import CommandLifecycle, CommandStage, LifecycleManager, STAGE_USER_LABELS
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
from integrations.app_catalog import AppCatalog
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

        # 1. State, Monitoring & IPC
        self.state_mgr = StateManager()
        self.status_broadcaster = StatusBroadcaster()
        self._overlay_proc: Optional[subprocess.Popen] = None
        self.lifecycle_mgr = LifecycleManager()
        self._command_lock = threading.Lock()
        self._is_processing = False
        self._active_lifecycle: Optional[CommandLifecycle] = None
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
        self.catalog = AppCatalog(cache_file=config.data_dir / "app_catalog.json")
        self.launcher = AppLauncher(cache_file=config.data_dir / "app_registry.json")
        self.atspi = AtspiClient()
        self.dbus = DBusClient()
        self.browser = BrowserClient()

        # 4. Tools Registry
        self.registry = ToolRegistry()
        self._register_all_tools()

        # 5. Execution & Verification
        self.binder = ParameterBinder(catalog=self.catalog)
        self.executor = ActionExecutor(self.registry, self.safety_policy, self.emergency_stop, config.safety)
        self.observer = DesktopObserver(self.kdotool, self.atspi)
        self.verifier = ActionVerifier(kdotool=self.kdotool)
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
            status_broadcaster=self.status_broadcaster,
            diagnostic_mode=diagnostic_mode,
            lifecycle_mgr=self.lifecycle_mgr,
        )

        # 7. Global Key Listener
        self.ptt_listener = PTTListener(
            shortcut=config.input.shortcut,
            ptt_key_name=config.input.ptt_key,
            estop_key_name=config.input.emergency_stop_key,
            activation_mode=config.input.activation_mode,
            on_press=self._on_ptt_press,
            on_release=self._on_ptt_release,
            on_estop=self._on_estop_press,
        )

    def _register_all_tools(self) -> None:
        self.registry.register(OpenApplicationTool(self.launcher, self.kdotool, catalog=self.catalog))
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
        """Called when user presses activation shortcut (e.g. Win + Alt)."""
        if self.emergency_stop.is_stopped():
            logger.warning("Ignoring shortcut: Agent is in Emergency Stop state.")
            return

        # Concurrency safety: if previous command still actively processing, avoid corruption
        if self._is_processing:
            logger.warning("Previous command still processing. Rejecting concurrent actuation.")
            return

        lifecycle = self.lifecycle_mgr.new_command()
        self._active_lifecycle = lifecycle
        lifecycle.transition(CommandStage.PTT_START)
        lifecycle.transition(CommandStage.AUDIO_START)

        logger.info("[%s] Voice Agent active: recording (%s)...", lifecycle.command_id, self.config.input.shortcut)
        self.state_mgr.set_state(AgentState.LISTENING, reason="Shortcut activated")
        self.status_broadcaster.broadcast(
            state="LISTENING",
            title="🎙 Listening",
            detail="Speak now",
            next_action="",
        )
        self.audio_capture.start()

    def _on_ptt_release(self) -> None:
        """Called when user releases Push-to-Talk key."""
        if not self.audio_capture.is_capturing:
            return

        lifecycle = self._active_lifecycle
        if not lifecycle:
            lifecycle = self.lifecycle_mgr.new_command()
            self._active_lifecycle = lifecycle

        lifecycle.transition(CommandStage.AUDIO_STOP)
        logger.info("[%s] Push-to-Talk released: finalizing speech with post-roll...", lifecycle.command_id)

        # Transition directly to TRANSCRIBING instead of generic PROCESSING
        lifecycle.transition(CommandStage.STT_START)
        self.status_broadcaster.broadcast(
            state="TRANSCRIBING",
            title="◌ Transcribing...",
            detail="Converting speech to text...",
            next_action="",
        )

        # Start global command watchdog (18.0 seconds limit)
        lifecycle.start_watchdog(timeout_sec=18.0, on_timeout=self._on_command_timeout)

        def _stop_and_process():
            with self._command_lock:
                self._is_processing = True
                try:
                    raw_pcm = self.audio_capture.stop(wait_postroll=True)
                    if not raw_pcm:
                        logger.warning("[%s] No audio captured on PTT release.", lifecycle.command_id)
                        lifecycle.fail(reason="No audio captured")
                        self.status_broadcaster.broadcast(
                            state="FAILURE",
                            title="● No speech detected",
                            detail="Try holding Meta + Alt while speaking",
                            next_action="",
                            level="warning",
                        )
                        self.state_mgr.set_state(AgentState.IDLE, reason="Empty audio")
                        if hasattr(self, "coordinator"):
                            from monitoring.latency_tracker import CommandMetrics
                            self.coordinator._speak_response("I didn't hear anything.", CommandMetrics(command_id=lifecycle.command_id))
                        return

                    self.coordinator.process_audio(raw_pcm, lifecycle=lifecycle)
                except Exception as e:
                    logger.exception("[%s] Unhandled exception in AudioProcessor: %s", lifecycle.command_id, e)
                    lifecycle.fail(reason=str(e))
                    self.status_broadcaster.broadcast(
                        state="FAILURE",
                        title="✕ Failed",
                        detail=f"Error: {str(e)[:30]}",
                        next_action="Try again",
                        level="error",
                    )
                    self.state_mgr.set_state(AgentState.IDLE, reason="Exception in processing")
                    if hasattr(self, "coordinator"):
                        from monitoring.latency_tracker import CommandMetrics
                        self.coordinator._speak_response("An error occurred.", CommandMetrics(command_id=lifecycle.command_id))
                finally:
                    lifecycle.stop_watchdog()
                    self._is_processing = False
                    self._active_lifecycle = None

        threading.Thread(
            target=_stop_and_process,
            daemon=True,
            name=f"AudioProc-{lifecycle.command_id}",
        ).start()

    def _on_command_timeout(self, lifecycle: CommandLifecycle) -> None:
        """Called automatically when a command exceeds its global watchdog timeout."""
        logger.error("[%s] Global watchdog timeout! Stage: %s", lifecycle.command_id, lifecycle.current_stage.value)
        if self.audio_capture.is_capturing:
            self.audio_capture.stop(wait_postroll=False)

        stuck_label = STAGE_USER_LABELS.get(lifecycle.current_stage, lifecycle.current_stage.value)
        self.status_broadcaster.broadcast(
            state="FAILURE",
            title="✕ Timed Out",
            detail=f"Stuck at: {stuck_label}",
            next_action="Hold Meta + Alt to try again",
            level="error",
        )
        self.state_mgr.set_state(AgentState.IDLE, reason="Command timeout")
        if hasattr(self, "coordinator"):
            from monitoring.latency_tracker import CommandMetrics
            self.coordinator._speak_response("Command timed out.", CommandMetrics(command_id=lifecycle.command_id))
        self._is_processing = False
        self._active_lifecycle = None

    def _on_estop_press(self) -> None:
        """Called when user hits Emergency Stop key."""
        self.emergency_stop.trigger(reason="Global hotkey F11 pressed")
        self.state_mgr.set_state(AgentState.PAUSED, reason="Emergency Stop")
        self.audio_capture.stop()
        self.status_broadcaster.broadcast(
            state="ERROR",
            title="⚠ EMERGENCY STOP",
            detail="Halted all pending actions",
            next_action="Hold Meta + Alt to resume",
            level="error",
        )
        self.dbus.show_notification("Laya Agent", "EMERGENCY STOP ACTIVATED", timeout_ms=4000)

    def start(self) -> None:
        """Start daemon and enter main loop."""
        if not self._acquire_single_instance_lock():
            sys.exit(1)

        logger.info("Starting Laya Voice Agent Daemon (Diagnostic Mode: %s)...", self.diagnostic_mode)

        # 0. Start Status IPC & Overlay
        self.status_broadcaster.start()
        try:
            overlay_env = os.environ.copy()
            overlay_env["LD_PRELOAD"] = "/usr/lib/libgtk4-layer-shell.so"
            self._overlay_proc = subprocess.Popen(
                [sys.executable, "-m", "app.ui.status_overlay"],
                env=overlay_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Spawned Live Status Overlay (PID: %d)", self._overlay_proc.pid)
        except Exception as e:
            logger.warning("Could not spawn status overlay: %s", e)

        # Setup OS signal handlers
        try:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGUSR1, self._handle_sigusr1)
            signal.signal(signal.SIGUSR2, self._handle_sigusr2)
        except (ValueError, AttributeError):
            logger.debug("Signal registration skipped (not in main thread)")

        # Pre-warm models & audio stream
        self.model_mgr.warm_models()
        self.model_mgr.start_sweeper()
        self.audio_capture.warm_stream()

        # Start PTT listener
        self.ptt_listener.start()

        self._running = True
        self.state_mgr.set_state(AgentState.IDLE, reason="Daemon started")
        self.status_broadcaster.broadcast(
            state="READY",
            title="",
            detail="",
            next_action="",
            level="info",
        )

        logger.info(
            "Laya Desktop Agent is running. Press %s to talk (or F12), F11 for Emergency Stop.",
            self.config.input.shortcut,
        )

        try:
            while self._running:
                time.sleep(1.0)
                # Periodic VRAM & health check
                self.vram_guard.check_status()
                # Check status overlay health
                if self._overlay_proc and self._overlay_proc.poll() is not None and self._running:
                    logger.warning("Status overlay terminated unexpectedly. Restarting...")
                    try:
                        overlay_env = os.environ.copy()
                        overlay_env["LD_PRELOAD"] = "/usr/lib/libgtk4-layer-shell.so"
                        self._overlay_proc = subprocess.Popen(
                            [sys.executable, "-m", "app.ui.status_overlay"],
                            env=overlay_env,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    except Exception as e:
                        logger.error("Failed to restart status overlay: %s", e)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _handle_sigusr1(self, signum, frame):
        logger.info("Received SIGUSR1: External activation trigger.")
        self.ptt_listener.trigger_activation()

    def _handle_sigusr2(self, signum, frame):
        logger.info("Received SIGUSR2: External VRAM cleanup trigger.")
        try:
            self.model_mgr.clean_vram()
            self.vram_guard.emergency_cleanup()
            logger.info("External VRAM cleanup completed.")
        except Exception as e:
            logger.warning("Error during SIGUSR2 VRAM cleanup: %s", e)

    def _handle_signal(self, signum, frame):
        logger.info("Received termination signal (%d). Shutting down cleanly...", signum)
        self._running = False

    def stop(self) -> None:
        """Clean shutdown of all subsystems."""
        if getattr(self, "_stopped", False):
            return
        self._stopped = True
        logger.info("Stopping Laya Desktop Agent...")
        self._running = False
        if self._overlay_proc:
            try:
                self._overlay_proc.terminate()
                self._overlay_proc.wait(timeout=2.0)
            except Exception:
                pass
            self._overlay_proc = None
        self.status_broadcaster.stop()
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
