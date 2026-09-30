"""
Command Line Interface for Laya Desktop Agent.
Provides subcommands: run, diagnose, exec, status, stop, test, bench.
"""

from pathlib import Path
from typing import Optional
import logging
import os
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


if __name__ == "__main__":
    cli()
