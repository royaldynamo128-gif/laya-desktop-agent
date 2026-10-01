"""
Physical Keyboard & Push-to-Talk Shortcut Diagnostic Tool.
Monitors all physical and virtual evdev keyboard devices in real time.
Explicitly traces keypress order, hold duration, release detection, and combo states.
"""

import logging
import signal
import sys
import time
import evdev
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.config import load_config
from app.ptt_listener import PTTListener

console = Console()


def run_physical_diagnostic(duration_sec: int = 30):
    config = load_config(profile="balanced")
    
    console.print(Panel.fit(
        f"[bold cyan]Physical Push-to-Talk Diagnostic Mode[/bold cyan]\n"
        f"Target Shortcut: [bold yellow]{config.input.shortcut}[/bold yellow] | Mode: [green]{config.input.activation_mode}[/green]\n"
        f"Hold [bold]{config.input.shortcut}[/bold] physically on your keyboard and observe state transitions below.\n"
        f"Test different orders: (Meta then Alt, Alt then Meta, release Alt first, release Meta first).\n"
        f"Auto-terminates after {duration_sec}s or press [bold red]Ctrl+C[/bold red].",
        title="Live Diagnostic Session",
    ))

    events_log = []
    combo_start_time = 0.0

    def on_raw_key(code: int, value: int, device_name: str):
        key_name = evdev.ecodes.KEY.get(code, str(code))
        action = "DOWN" if value == 1 else ("REPEAT" if value == 2 else "UP")
        timestamp = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
        
        # Filter noise, show modifiers and functional keys
        if "META" in key_name or "ALT" in key_name or "KEY_F1" in key_name or "KEY_F12" in key_name or "KEY_F11" in key_name:
            style = "bold yellow" if action == "DOWN" else ("dim" if action == "REPEAT" else "cyan")
            console.print(f"[{timestamp}] [{style}]{action:6}[/{style}] {key_name:18} (code: {code:3}) on [dim]{device_name}[/dim]")

    def on_press():
        nonlocal combo_start_time
        combo_start_time = time.perf_counter()
        timestamp = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
        console.print(f"\n[bold green]>>> [{timestamp}] PTT: COMBINATION ACTIVE ({config.input.shortcut}) -> RECORDING STARTED <<<[/bold green]\n")
        events_log.append({"event": "PRESS", "time": time.time()})

    def on_release():
        nonlocal combo_start_time
        duration = time.perf_counter() - combo_start_time if combo_start_time else 0.0
        timestamp = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
        console.print(f"\n[bold magenta]>>> [{timestamp}] PTT: KEY RELEASE DETECTED -> RECORDING STOPPED (Held: {duration:.3f}s) <<<[/bold magenta]\n")
        events_log.append({"event": "RELEASE", "duration": duration, "time": time.time()})

    def on_estop():
        timestamp = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
        console.print(f"\n[bold red]>>> [{timestamp}] PTT: EMERGENCY STOP TRIGGERED (F11) <<<[/bold red]\n")
        events_log.append({"event": "ESTOP", "time": time.time()})

    listener = PTTListener(
        shortcut=config.input.shortcut,
        ptt_key_name=config.input.ptt_key,
        estop_key_name=config.input.emergency_stop_key,
        activation_mode=config.input.activation_mode,
        on_press=on_press,
        on_release=on_release,
        on_estop=on_estop,
        on_raw_event=on_raw_key,
    )

    devices = listener.find_keyboard_devices()
    table = Table(title="Discovered Keyboard Input Devices")
    table.add_column("Device Node", style="cyan")
    table.add_column("Device Name", style="bold")
    for dev in devices:
        table.add_row(dev.path, dev.name)
    console.print(table)

    if not listener.start():
        console.print("[bold red]Failed to start evdev listener. Check /dev/input permissions.[/bold red]")
        return False

    console.print("\n[bold green]Listener ACTIVE. Waiting for physical keyboard input...[/bold green]\n")

    start_t = time.time()
    try:
        while time.time() - start_t < duration_sec:
            time.sleep(0.1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Test interrupted by user.[/yellow]")
    finally:
        listener.stop()

    press_count = sum(1 for e in events_log if e["event"] == "PRESS")
    release_count = sum(1 for e in events_log if e["event"] == "RELEASE")
    
    console.print(Panel.fit(
        f"Total Combination Press Events: [bold]{press_count}[/bold]\n"
        f"Total Release Events: [bold]{release_count}[/bold]\n"
        f"Status: {'[bold green]PASSED - Physical Key Events Verified[/bold green]' if press_count > 0 else '[yellow]NO EVENTS DETECTED - Please verify you pressed the keys[/yellow]'}",
        title="Physical Shortcut Diagnostic Summary",
    ))
    return press_count > 0


if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    run_physical_diagnostic(duration_sec=duration)
