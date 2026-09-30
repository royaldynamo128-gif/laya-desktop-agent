"""
Multi-App Live Verification for Win + Alt Global Shortcut.
Tests real hardware scancode injection via Linux kernel /dev/uinput
while actively switching focus between real running desktop applications (Chrome, Dolphin, Terminal).
"""

import os
import sys
import time
import threading
from typing import Dict, List, Tuple
import evdev
from evdev import UInput, ecodes

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import load_config
from core.state import AgentState
from app.daemon import LayaDaemon
from integrations.kdotool_client import KdotoolClient
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def discover_open_apps(kdotool: KdotoolClient) -> List[Tuple[str, str]]:
    """Discover active application windows."""
    windows = []
    candidates = kdotool.search("", limit=20)
    for win_id in candidates:
        name = kdotool.get_window_name(win_id)
        if name and not name.startswith("Desktop") and name.strip():
            windows.append((win_id, name.strip()))
    return windows


def simulate_key_combo(v_kbd: UInput, keys: List[int], hold_duration: float = 0.4):
    """Simulate physical hardware press and release of key combination."""
    for k in keys:
        v_kbd.write(ecodes.EV_KEY, k, 1)
        v_kbd.syn()
        time.sleep(0.02)

    time.sleep(hold_duration)

    for k in reversed(keys):
        v_kbd.write(ecodes.EV_KEY, k, 0)
        v_kbd.syn()
        time.sleep(0.02)


def wait_for_idle(daemon: LayaDaemon, timeout: float = 4.0) -> bool:
    """Wait for daemon to return to clean IDLE state after processing."""
    start = time.time()
    while time.time() - start < timeout:
        if daemon.state_mgr.current_state == AgentState.IDLE and not daemon.audio_capture.is_capturing:
            return True
        time.sleep(0.1)
    daemon.state_mgr.set_state(AgentState.IDLE)
    daemon.ptt_listener.reset()
    return False


def run_multi_app_verification():
    console.print(Panel("[bold green]Starting Multi-App Win + Alt Live Desktop Test[/bold green]", expand=False))

    kdotool = KdotoolClient()
    initial_active_win = kdotool.get_active_window()
    initial_win_name = kdotool.get_window_name(initial_active_win) if initial_active_win else "agy"
    console.print(f"[bold cyan]Current Active Window:[/bold cyan] {initial_win_name} ({initial_active_win})")

    # Discover open windows
    open_apps = discover_open_apps(kdotool)
    console.print(f"[cyan]Discovered {len(open_apps)} application windows on KDE Wayland:[/cyan]")
    for wid, name in open_apps:
        console.print(f"  • [bold]{name}[/bold] (ID: {wid})")

    # Create kernel /dev/uinput virtual hardware keyboard
    console.print("\n[cyan]Initializing virtual hardware keyboard via /dev/uinput...[/cyan]")
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
    v_kbd = UInput(cap, name="LayaMultiAppTestKeyboard")
    time.sleep(0.5)
    console.print(f"[green]Virtual keyboard active at: {v_kbd.device.path}[/green]")

    # Start LayaDaemon in diagnostic mode
    cfg = load_config(profile="balanced")
    console.print(f"[cyan]Loaded configuration: shortcut=[bold]{cfg.input.shortcut}[/bold], mode=[bold]{cfg.input.activation_mode}[/bold][/cyan]")

    daemon = LayaDaemon(cfg, diagnostic_mode=True)
    daemon_thread = threading.Thread(target=daemon.start, daemon=True, name="DaemonMultiAppThread")
    daemon_thread.start()

    console.print("[cyan]Waiting for daemon initialization...[/cyan]")
    ready = False
    for _ in range(30):
        time.sleep(0.5)
        if daemon._running:
            ready = True
            break

    if not ready:
        console.print("[red]Daemon failed to initialize. Aborting.[/red]")
        v_kbd.close()
        sys.exit(1)

    console.print(f"[bold green]Daemon ONLINE & READY in state: {daemon.state_mgr.current_state.value}[/bold green]\n")

    results = []

    try:
        # Test each discovered window
        for win_id, win_name in open_apps:
            console.print(f"[bold magenta]Testing Win + Alt in window: {win_name}[/bold magenta]")
            
            # 1. Activate window
            kdotool.activate_window(win_id)
            time.sleep(0.4)
            active_now = kdotool.get_active_window()
            active_name_now = kdotool.get_window_name(active_now) if active_now else "Unknown"
            console.print(f"  Focused: [yellow]{active_name_now}[/yellow] (ID: {active_now})")

            # Ensure clean IDLE state
            wait_for_idle(daemon, timeout=2.0)
            daemon.ptt_listener.reset()
            time.sleep(0.2)

            # 2. Push-to-Talk Test (Hold 400ms)
            console.print("  [cyan]-> Testing Push-to-Talk (Hold Win + Alt)...[/cyan]")
            for k in [ecodes.KEY_LEFTMETA, ecodes.KEY_LEFTALT]:
                v_kbd.write(ecodes.EV_KEY, k, 1)
                v_kbd.syn()
                time.sleep(0.02)

            # Check transition to LISTENING
            ptt_listening = False
            for _ in range(25):
                time.sleep(0.05)
                if daemon.state_mgr.current_state == AgentState.LISTENING:
                    ptt_listening = True
                    break

            # Hold for 400ms
            time.sleep(0.4)

            # Release
            for k in [ecodes.KEY_LEFTALT, ecodes.KEY_LEFTMETA]:
                v_kbd.write(ecodes.EV_KEY, k, 0)
                v_kbd.syn()
                time.sleep(0.02)

            # Check transition to PROCESSING or IDLE
            ptt_processed = False
            for _ in range(25):
                time.sleep(0.05)
                st = daemon.state_mgr.current_state
                if st in (AgentState.PROCESSING, AgentState.IDLE):
                    ptt_processed = True
                    break

            ptt_ok = ptt_listening and ptt_processed
            console.print(f"     PTT Listening: {'✓' if ptt_listening else '✗'} | PTT Release/Process: {'✓' if ptt_processed else '✗'}")

            # Wait for clean IDLE before next mode
            wait_for_idle(daemon, timeout=3.0)
            daemon.ptt_listener.reset()
            time.sleep(0.3)

            # 3. Toggle Mode Test (Quick Tap < 200ms)
            console.print("  [cyan]-> Testing Toggle Mode (Quick Tap Win + Alt)...[/cyan]")
            # Quick Tap Press
            for k in [ecodes.KEY_LEFTMETA, ecodes.KEY_LEFTALT]:
                v_kbd.write(ecodes.EV_KEY, k, 1)
                v_kbd.syn()
            time.sleep(0.05)
            # Quick Tap Release
            for k in [ecodes.KEY_LEFTALT, ecodes.KEY_LEFTMETA]:
                v_kbd.write(ecodes.EV_KEY, k, 0)
                v_kbd.syn()

            toggle_started = False
            for _ in range(25):
                time.sleep(0.05)
                if daemon.state_mgr.current_state == AgentState.LISTENING:
                    toggle_started = True
                    break

            time.sleep(0.3)

            # Second tap to deactivate toggle
            for k in [ecodes.KEY_LEFTMETA, ecodes.KEY_LEFTALT]:
                v_kbd.write(ecodes.EV_KEY, k, 1)
                v_kbd.syn()
            time.sleep(0.05)
            for k in [ecodes.KEY_LEFTALT, ecodes.KEY_LEFTMETA]:
                v_kbd.write(ecodes.EV_KEY, k, 0)
                v_kbd.syn()

            toggle_stopped = False
            for _ in range(25):
                time.sleep(0.05)
                st = daemon.state_mgr.current_state
                if st in (AgentState.PROCESSING, AgentState.IDLE):
                    toggle_stopped = True
                    break

            toggle_ok = toggle_started and toggle_stopped
            console.print(f"     Toggle Start: {'✓' if toggle_started else '✗'} | Toggle Stop: {'✓' if toggle_stopped else '✗'}")

            # Settle after toggle
            wait_for_idle(daemon, timeout=3.0)
            daemon.ptt_listener.reset()

            results.append({
                "app": win_name,
                "window_id": win_id,
                "ptt_ok": ptt_ok,
                "toggle_ok": toggle_ok,
                "overall": ptt_ok and toggle_ok,
            })
            console.print()

    finally:
        # Restore initial active window
        if initial_active_win:
            console.print(f"[cyan]Restoring active window to: {initial_win_name}...[/cyan]")
            kdotool.activate_window(initial_active_win)

        # Cleanup
        console.print("[cyan]Shutting down virtual device and daemon...[/cyan]")
        try:
            v_kbd.close()
        except Exception:
            pass
        try:
            daemon.stop()
        except Exception:
            pass

    # Print summary table
    table = Table(title="Win + Alt Global Shortcut Across Desktop Applications")
    table.add_column("Application Window", style="bold")
    table.add_column("Push-to-Talk (Hold >350ms)")
    table.add_column("Toggle Mode (Tap <350ms)")
    table.add_column("Result", style="bold")

    all_passed = True
    for r in results:
        status_str = "[bold green]PASS[/bold green]" if r["overall"] else "[bold red]FAIL[/bold red]"
        if not r["overall"]:
            all_passed = False
        table.add_row(
            r["app"][:40],
            "[green]PASSED[/green]" if r["ptt_ok"] else "[red]FAILED[/red]",
            "[green]PASSED[/green]" if r["toggle_ok"] else "[red]FAILED[/red]",
            status_str,
        )

    console.print(table)
    if all_passed and results:
        console.print(Panel("[bold green]ALL MULTI-APP TESTS PASSED! Win + Alt works reliably in every app.[/bold green]"))
    else:
        console.print(Panel("[bold red]SOME APP TESTS FAILED.[/bold red]"))
        sys.exit(1)


if __name__ == "__main__":
    run_multi_app_verification()
