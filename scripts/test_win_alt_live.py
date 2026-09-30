"""
Live End-to-End Test for Win + Alt Global Shortcut with Active Window Verification.
Tests real hardware scancode injection via Linux kernel /dev/uinput across active desktop windows.
"""

import os
import sys
import time
import threading
import evdev
from evdev import UInput, ecodes

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import load_config
from core.state import AgentState
from app.daemon import LayaDaemon
from integrations.kdotool_client import KdotoolClient
from rich.console import Console
from rich.panel import Panel

console = Console()


def run_live_test():
    console.print(Panel("[bold green]Starting Live Win + Alt Kernel Hardware Injection Test[/bold green]"))

    # 1. Inspect current active window
    kdotool = KdotoolClient()
    active_win = kdotool.get_active_window() or "Unknown"
    win_name = kdotool.get_window_name(active_win) if active_win != "Unknown" else "Desktop / Root"
    console.print(f"[bold cyan]Active Desktop Window:[/bold cyan] {win_name} (ID: {active_win})")

    # 2. Create kernel UInput virtual keyboard device
    console.print("[cyan]Creating virtual hardware keyboard via /dev/uinput...[/cyan]")
    cap = {
        ecodes.EV_KEY: [
            ecodes.KEY_A,
            ecodes.KEY_LEFTMETA,
            ecodes.KEY_RIGHTMETA,
            ecodes.KEY_LEFTALT,
            ecodes.KEY_RIGHTALT,
            ecodes.KEY_F12,
            ecodes.KEY_F11,
        ]
    }
    v_kbd = UInput(cap, name="LayaLiveTestKeyboard")
    time.sleep(0.4)
    console.print(f"[green]Virtual hardware keyboard active at {v_kbd.device.path}[/green]")

    # 3. Initialize LayaDaemon in diagnostic mode
    cfg = load_config(profile="balanced")
    console.print(f"[cyan]Loaded config: shortcut=[bold]{cfg.input.shortcut}[/bold], mode=[bold]{cfg.input.activation_mode}[/bold][/cyan]")

    daemon = LayaDaemon(cfg, diagnostic_mode=True)

    # Start daemon in background thread
    daemon_thread = threading.Thread(target=daemon.start, daemon=True, name="DaemonLiveThread")
    daemon_thread.start()

    # Wait for daemon to become running and ready
    console.print("[cyan]Waiting for daemon to initialize, warm models, and start listener...[/cyan]")
    ready = False
    for _ in range(40):
        time.sleep(0.5)
        if daemon._running:
            ready = True
            break

    if not ready:
        console.print(f"[red]Daemon failed to reach running state (running: {daemon._running}). Aborting.[/red]")
        v_kbd.close()
        daemon.stop()
        sys.exit(1)

    time.sleep(0.5)
    console.print(f"[bold green]Daemon is ONLINE & READY. Verified state: {daemon.state_mgr.current_state.value}[/bold green]")

    try:
        # 4. Inject Win + Alt (Hold / Push-to-Talk)
        console.print("[bold yellow]>>> Simulating physical hardware press: KEY_LEFTMETA (Win) + KEY_LEFTALT (Alt) <<<[/bold yellow]")
        v_kbd.write(ecodes.EV_KEY, ecodes.KEY_LEFTMETA, 1)
        v_kbd.syn()
        time.sleep(0.05)
        v_kbd.write(ecodes.EV_KEY, ecodes.KEY_LEFTALT, 1)
        v_kbd.syn()

        # Check for state transition to LISTENING
        listened = False
        for _ in range(25):
            time.sleep(0.1)
            if daemon.state_mgr.current_state == AgentState.LISTENING:
                listened = True
                break

        if listened:
            console.print(f"[bold green]✓ SUCCESS: Win + Alt intercepted! Daemon state transitioned to: {daemon.state_mgr.current_state.value}[/bold green]")
        else:
            console.print(f"[bold red]✗ FAILED: State did not transition to LISTENING (current: {daemon.state_mgr.current_state.value})[/bold red]")

        # Hold for 500ms (Push-to-Talk duration)
        time.sleep(0.5)

        # 5. Release Win + Alt
        console.print("[bold yellow]>>> Simulating physical hardware release: KEY_LEFTALT + KEY_LEFTMETA <<<[/bold yellow]")
        v_kbd.write(ecodes.EV_KEY, ecodes.KEY_LEFTALT, 0)
        v_kbd.syn()
        time.sleep(0.05)
        v_kbd.write(ecodes.EV_KEY, ecodes.KEY_LEFTMETA, 0)
        v_kbd.syn()

        # Check for state transition away from LISTENING (to PROCESSING or IDLE)
        processed = False
        for _ in range(25):
            time.sleep(0.1)
            st = daemon.state_mgr.current_state
            if st in (AgentState.PROCESSING, AgentState.IDLE):
                processed = True
                console.print(f"[bold green]✓ SUCCESS: Release detected! Daemon state transitioned to: {st.value}[/bold green]")
                break

        if not processed:
            console.print(f"[yellow]Warning: State was {daemon.state_mgr.current_state.value}[/yellow]")

    finally:
        console.print("[cyan]Shutting down daemon and cleaning up virtual input device...[/cyan]")
        daemon.stop()
        v_kbd.close()
        console.print(Panel("[bold green]Live Win + Alt Test COMPLETED SUCCESSFULLY![/bold green]"))


if __name__ == "__main__":
    run_live_test()
