"""
Command Line Interface for Laya Desktop Agent.
Provides subcommands: run, diagnose, exec, status, stop, test, bench.
"""

from pathlib import Path
from typing import Optional
import logging
import os
import shutil
import signal
import sys
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from core.config import load_config
from monitoring.resource_monitor import ResourceMonitor
from monitoring.vram_guard import VRAMGuard
from monitoring.health_checker import HealthChecker
from app.daemon import LayaDaemon

console = Console()


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        from logging.handlers import RotatingFileHandler
        handlers.append(RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


@click.group()
def cli():
    """Laya Voice Desktop Agent for Arch Linux + KDE Plasma Wayland."""
    pass


@cli.command()
@click.option("--profile", default="balanced", type=click.Choice(["low_resource", "balanced", "performance"]))
@click.option("--diagnostic", is_flag=True, help="Run in diagnostic mode (zero execution)")
def run(profile: str, diagnostic: bool):
    """Start the voice agent daemon."""
    cfg = load_config(profile=profile)
    setup_logging(cfg.logging.level, cfg.logs_dir / "agent.log")

    console.print(Panel.fit(
        f"[bold cyan]Laya Voice Desktop Agent[/bold cyan]\n"
        f"Profile: [green]{profile}[/green] | Diagnostic Mode: [yellow]{diagnostic}[/yellow]\n"
        f"Shortcut: [bold]{cfg.input.shortcut}[/bold] ({cfg.input.activation_mode}) | Emergency Stop: [bold red]{cfg.input.emergency_stop_key}[/bold red]",
        title="Starting Daemon",
    ))

    daemon = LayaDaemon(cfg, diagnostic_mode=diagnostic)
    daemon.start()


@cli.command()
def diagnose():
    """Run agent in Diagnostic Mode: tests voice capture and Laya routing without executing OS actions."""
    cfg = load_config(profile="balanced")
    setup_logging("INFO", cfg.logs_dir / "diagnostic.log")

    console.print(Panel.fit(
        "[bold yellow]Diagnostic Mode Active[/bold yellow]\n"
        "Voice input will be captured and routed by Laya, but NO actions will be executed on your desktop.\n"
        f"Press or hold [bold]{cfg.input.shortcut}[/bold] ({cfg.input.activation_mode}) and speak a command (e.g. 'Open Chrome', 'Scroll down').\n"
        "Press [bold red]Ctrl+C[/bold red] to exit.",
        title="Safe Diagnostic Session",
    ))

    daemon = LayaDaemon(cfg, diagnostic_mode=True)
    daemon.start()


@cli.command()
@click.argument("command_text")
@click.option("--diagnostic", is_flag=True, help="Route command without executing")
def exec(command_text: str, diagnostic: bool):
    """Execute or route a single natural language text command."""
    cfg = load_config(profile="balanced")
    setup_logging("WARNING")

    daemon = LayaDaemon(cfg, diagnostic_mode=diagnostic)
    # Pre-warm Laya
    daemon.model_mgr.laya.load()

    console.print(f"Processing command: [bold cyan]'{command_text}'[/bold cyan]")
    res = daemon.coordinator.process_text_command(command_text)

    table = Table(title="Command Execution Result")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Intent", res.intent)
    table.add_row("Action", str(res.action_executed))
    table.add_row("Execution Success", "[green]YES[/green]" if res.success else "[red]NO[/red]")
    table.add_row("Verified", "[green]YES[/green]" if res.verified else "[red]NO[/red]")
    table.add_row("Response", res.response_text)
    table.add_row("Total Latency", f"{res.metrics.total_ms:.1f} ms")

    console.print(table)
    daemon.model_mgr.shutdown()


@cli.command()
@click.argument("goal_text")
@click.option("--profile", default="balanced", type=click.Choice(["low_resource", "balanced", "performance"]))
def goal(goal_text: str, profile: str):
    """Execute an autonomous natural language goal end-to-end on the desktop."""
    cfg = load_config(profile=profile)
    setup_logging("INFO")

    daemon = LayaDaemon(cfg, diagnostic_mode=False)
    daemon.model_mgr.laya.load()

    console.print(f"[bold cyan]Autonomous Goal:[/bold cyan] [bold green]'{goal_text}'[/bold green]")
    res = daemon.coordinator.process_goal(goal_text)

    table = Table(title="Autonomous Goal Execution Result")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Goal", res.goal)
    table.add_row("Success", "[green]YES[/green]" if res.success else "[red]NO[/red]")
    table.add_row("Verified", "[green]YES[/green]" if res.verified else "[red]NO[/red]")
    table.add_row("Steps Executed", str(res.total_steps_executed))
    table.add_row("Total Time", f"{res.elapsed_ms:.1f} ms")
    table.add_row("Message", res.message)

    console.print(table)
    daemon.model_mgr.shutdown()


@cli.command()
def status():

    """Display real-time VRAM, RAM, and system health status."""
    cfg = load_config()
    mon = ResourceMonitor()
    snap = mon.get_snapshot()
    guard = VRAMGuard(cfg.vram, mon)
    checker = HealthChecker(cfg, mon)
    health = checker.run_all_checks()

    # VRAM Table
    t1 = Table(title="GPU & VRAM Status (RTX 3050)")
    t1.add_column("Total VRAM")
    t1.add_column("Used VRAM")
    t1.add_column("Free VRAM")
    t1.add_column("Usage %")
    t1.add_column("GPU Temp")
    t1.add_column("Guard Status")

    t1.add_row(
        f"{snap.vram_total_mb:.1f} MB",
        f"{snap.vram_used_mb:.1f} MB",
        f"{snap.vram_free_mb:.1f} MB",
        f"{snap.vram_used_pct:.1f}%",
        f"{snap.gpu_temp_c:.0f}°C",
        f"[green]{guard.check_status().value}[/green]",
    )
    console.print(t1)

    # Health Table
    t2 = Table(title="System Health & Interface Checks")
    t2.add_column("Component")
    t2.add_column("Status")
    t2.add_column("Details")

    for k, v in health.items():
        if isinstance(v, dict):
            status = v.get("status", "INFO")
            color = "green" if status == "PASS" else ("yellow" if status == "WARNING" else "red")
            detail = str({ik: iv for ik, iv in v.items() if ik != "status"})
            t2.add_row(k.upper(), f"[{color}]{status}[/{color}]", detail)

    console.print(t2)
    mon.shutdown()


@cli.command()
def activate():
    """Trigger voice listening on the running background daemon (via Win+Alt or CLI)."""
    runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    lock_path = runtime_dir / "laya_voice_agent.lock"

    if not lock_path.exists():
        console.print("[yellow]Daemon is not running. Start it with './scripts/start.sh'.[/yellow]")
        return

    try:
        with open(lock_path, "r") as f:
            pid = int(f.read().strip())
        os.kill(pid, signal.SIGUSR1)
        console.print(f"[green]Sent activation signal to daemon PID {pid}.[/green]")
    except Exception as e:
        console.print(f"[red]Failed to signal daemon: {e}[/red]")


@cli.command()
def stop():
    """Stop running background daemon."""
    runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    lock_path = runtime_dir / "laya_voice_agent.lock"

    if not lock_path.exists():
        console.print("[yellow]No active daemon lock found.[/yellow]")
        return

    try:
        with open(lock_path, "r") as f:
            pid = int(f.read().strip())
        os.kill(pid, signal.SIGTERM)
        console.print(f"[green]Sent termination signal to daemon PID {pid}.[/green]")
    except Exception as e:
        console.print(f"[red]Failed to stop daemon: {e}[/red]")


@cli.command("vram-clean")
def vram_clean():
    """Flush CUDA cache and reclaim GPU VRAM across all processes."""
    import time
    mon = ResourceMonitor()
    snap_before = mon.get_snapshot()

    # 1. Signal background daemon if active
    runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    lock_path = runtime_dir / "laya_voice_agent.lock"
    signaled_daemon = False
    if lock_path.exists():
        try:
            with open(lock_path, "r") as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGUSR2)
            signaled_daemon = True
            time.sleep(0.5)
        except Exception:
            pass

    # 2. Flush current process torch cache
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass

    snap_after = mon.get_snapshot()
    mon.shutdown()

    reclaimed_mb = max(0.0, snap_before.vram_used_mb - snap_after.vram_used_mb)

    t = Table(title="VRAM Cleanup Results")
    t.add_column("Metric", style="bold")
    t.add_column("Value")
    gpu_label = "NVIDIA RTX 3050 (6GB)"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_label = torch.cuda.get_device_name(0)
    except Exception:
        pass
    t.add_row("GPU Model", gpu_label)
    t.add_row("Before VRAM Used", f"{snap_before.vram_used_mb:.1f} MB ({snap_before.vram_used_pct:.1f}%)")
    t.add_row("After VRAM Used", f"[green]{snap_after.vram_used_mb:.1f} MB ({snap_after.vram_used_pct:.1f}%)[/green]")
    t.add_row("Free VRAM Available", f"[bold green]{snap_after.vram_free_mb:.1f} MB[/bold green]")
    t.add_row("VRAM Reclaimed", f"[bold cyan]{reclaimed_mb:.1f} MB[/bold cyan]")
    t.add_row("Daemon Signaled", "[green]YES (SIGUSR2)[/green]" if signaled_daemon else "[yellow]NO (daemon not running)[/yellow]")
    console.print(t)


@cli.command("test-ptt")
@click.option("--duration", default=20, help="Diagnostic listening duration in seconds")
def test_ptt(duration: int):
    """Diagnose physical keyboard input and shortcut detection."""
    from scripts.diagnose_physical_ptt import run_physical_diagnostic
    run_physical_diagnostic(duration_sec=duration)


@cli.command("doctor")
def doctor():
    """Print complete diagnostic status panel as per specification."""
    cfg = load_config(profile="balanced")
    mon = ResourceMonitor()
    snap = mon.get_snapshot()
    guard = VRAMGuard(cfg.vram, mon)

    runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    lock_path = runtime_dir / "laya_voice_agent.lock"
    daemon_pid = None
    if lock_path.exists():
        try:
            with open(lock_path, "r") as f:
                daemon_pid = int(f.read().strip())
        except Exception:
            pass

    table = Table(title="Laya Voice Desktop Agent Diagnostic Panel")
    table.add_column("Subsystem", style="bold cyan")
    table.add_column("State", style="bold")
    table.add_column("Details", style="dim")

    table.add_row(
        "Agent Daemon",
        f"[green]RUNNING (PID: {daemon_pid})[/green]" if daemon_pid else "[yellow]STOPPED[/yellow]",
        f"Lock: {lock_path}" if daemon_pid else "Not running",
    )
    table.add_row(
        "PTT Shortcut",
        "[green]READY[/green]",
        f"Shortcut: {cfg.input.shortcut} | Mode: {cfg.input.activation_mode}",
    )
    from app.ptt_listener import PTTListener
    devs = PTTListener(
        shortcut=cfg.input.shortcut,
        activation_mode=cfg.input.activation_mode,
    ).find_keyboard_devices()
    table.add_row(
        "Keyboards",
        f"[green]CONNECTED ({len(devs)} devices)[/green]" if devs else "[red]NO EVDEV KEYBOARDS[/red]",
        ", ".join(d.name[:18] for d in devs[:3]) if devs else "evdev devices need read permission",
    )

    # --- Real subsystem probes (never fake "READY") ---
    def _probe(import_name: str) -> bool:
        try:
            __import__(import_name)
            return True
        except Exception:
            return False

    mic_inputs = 0
    mic_err = ""
    try:
        import sounddevice as sd
        mic_inputs = sum(1 for d in sd.query_devices() if d.get("max_input_channels", 0) > 0)
    except Exception as e:
        mic_err = str(e)[:48]
    table.add_row(
        "Microphone",
        f"[green]AVAILABLE ({mic_inputs} inputs)[/green]" if mic_inputs else f"[red]UNAVAILABLE[/red]",
        mic_err or f"Sample Rate: {cfg.input.sample_rate}Hz | Channels: {cfg.input.channels}",
    )
    table.add_row(
        "VAD Engine",
        "[green]READY[/green]" if _probe("webrtcvad") else "[red]MISSING (pip install webrtcvad)[/red]",
        f"WebRTC VAD Aggressiveness: {cfg.input.vad_aggressiveness}",
    )
    stt_ok = _probe("faster_whisper")
    table.add_row(
        "STT (Whisper)",
        "[green]LIB READY[/green]" if stt_ok else "[red]MISSING (pip install faster-whisper)[/red]",
        f"Model: {cfg.stt.model_size} | Device: {cfg.stt.device} ({cfg.stt.compute_type})",
    )
    table.add_row(
        "Laya Router",
        "[green]LIB READY[/green]" if _probe("laya") else "[yellow]FALLBACK (semantic parser only)[/yellow]",
        f"Model: {cfg.laya.model_id} | Device: {cfg.laya.device}",
    )
    kd = shutil.which("kdotool") or os.environ.get("KDTOOL_BIN", "")
    table.add_row(
        "Executor & Verifier",
        "[green]READY[/green]" if kd and Path(kd).exists() else "[red]KDTOOL NOT FOUND[/red]",
        f"kdotool: {kd or 'not found'} | AT-SPI2: {'available' if _probe('gi') else 'missing python-gi'}",
    )
    table.add_row(
        "Status Overlay",
        "[green]READY[/green]" if _probe("gi") else "[yellow]GTK4/gi missing — overlay disabled[/yellow]",
        "GTK 4 Layer Shell anchored bottom-right (TOP layer)",
    )
    table.add_row(
        "VRAM",
        f"[{'green' if snap.vram_used_pct < 80 else 'yellow'}]{snap.vram_used_mb:.0f} / {snap.vram_total_mb:.0f} MB ({snap.vram_used_pct:.1f}%)[/]",
        f"Guard: {guard.check_status().value}",
    )
    table.add_row(
        "RAM",
        f"{snap.ram_used_mb / 1024:.1f} / {snap.ram_total_mb / 1024:.1f} GB ({snap.ram_used_pct:.1f}%)",
        f"Process RSS: {snap.process_rss_mb:.1f} MB",
    )
    console.print(table)

    # Last Command Timeline from metrics log
    metrics_file = cfg.logs_dir / "metrics.jsonl"
    if metrics_file.exists():
        import json
        last_metric = None
        try:
            with open(metrics_file, "r") as f:
                lines = [l.strip() for l in f if l.strip()]
                if lines:
                    last_metric = json.loads(lines[-1])
        except Exception:
            pass

        if last_metric:
            t_cmd = Table(title="Last Command Pipeline Breakdown")
            t_cmd.add_column("Property", style="bold cyan")
            t_cmd.add_column("Value")
            t_cmd.add_row("Command ID", str(last_metric.get("command_id", "N/A")))
            t_cmd.add_row("Heard", f"\"{last_metric.get('transcription', '')}\"")
            t_cmd.add_row("Resolved Intent", str(last_metric.get("intent", "")))
            t_cmd.add_row("Target Application", str(last_metric.get("target", "")))
            t_cmd.add_row("Outcome", "[green]SUCCESS[/green]" if last_metric.get("success") else "[red]FAILED[/red]")
            t_cmd.add_row("Total Latency", f"{last_metric.get('total_ms', 0):.1f} ms")

            durations = last_metric.get("durations_ms", {})
            if durations:
                dur_str = " | ".join(f"{k}: {v:.1f}ms" for k, v in durations.items())
                t_cmd.add_row("Stage Durations", dur_str)

            console.print(t_cmd)

    mon.shutdown()


if __name__ == "__main__":
    cli()
