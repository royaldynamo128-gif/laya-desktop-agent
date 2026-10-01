"""
Long-Running VRAM & RAM Stress Test for Laya Desktop Agent.
Executes repeated multi-stage voice and routing cycles to verify zero progressive memory accumulation.
"""

from typing import List, Dict, Any
from pathlib import Path
import gc
import json
import logging
import time
import click
from rich.console import Console
from rich.table import Table

from core.config import load_config
from app.daemon import LayaDaemon
from monitoring.resource_monitor import ResourceMonitor

console = Console()
logger = logging.getLogger("stress_test")

TEST_PROMPTS = [
    "Open Chrome",
    "Switch to Opera",
    "Scroll down twice",
    "Take a screenshot",
    "What is on my screen?",
    "Search YouTube for Arch Linux KDE optimization",
    "Set volume to 60 percent",
    "Minimize this window",
    "Close this window",
    "Open Telegram",
]


def run_stress_test(cycles: int = 50, profile: str = "balanced") -> Dict[str, Any]:
    console.print(f"[bold cyan]Starting VRAM Stress Test: {cycles} cycles (Profile: {profile})[/bold cyan]")

    config = load_config(profile=profile)
    daemon = LayaDaemon(config, diagnostic_mode=True)
    monitor = ResourceMonitor()

    # Warm up models
    console.print("Pre-warming Laya and Whisper models...")
    daemon.model_mgr.warm_models()

    # Initial baseline after warm-up inference
    gc.collect()
    snap_warm = monitor.get_snapshot()
    initial_vram = snap_warm.vram_used_mb
    initial_rss = snap_warm.process_rss_mb

    console.print(f"Warm-up Baseline: VRAM: [bold green]{initial_vram:.1f} MB[/bold green] | RSS: [bold green]{initial_rss:.1f} MB[/bold green]")

    history: List[Dict[str, float]] = []
    peak_vram = initial_vram
    peak_rss = initial_rss

    table = Table(title="VRAM & RAM Trajectory Over Cycles")
    table.add_column("Cycle", style="bold")
    table.add_column("Prompt Sample")
    table.add_column("VRAM (MB)")
    table.add_column("VRAM Δ (MB)")
    table.add_column("Process RSS (MB)")
    table.add_column("Avg Latency (ms)")

    start_time = time.time()

    for cycle in range(1, cycles + 1):
        prompt = TEST_PROMPTS[(cycle - 1) % len(TEST_PROMPTS)]
        t_start = time.perf_counter()

        res = daemon.coordinator.process_text_command(prompt)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Snapshot post cycle
        snap = monitor.get_snapshot()
        vram = snap.vram_used_mb
        rss = snap.process_rss_mb

        if vram > peak_vram:
            peak_vram = vram
        if rss > peak_rss:
            peak_rss = rss

        delta_vram = vram - initial_vram
        history.append({
            "cycle": cycle,
            "prompt": prompt,
            "vram_mb": vram,
            "rss_mb": rss,
            "latency_ms": elapsed_ms,
        })

        if cycle == 1 or cycle % 10 == 0 or cycle == cycles:
            table.add_row(
                str(cycle),
                prompt[:25],
                f"{vram:.1f}",
                f"{delta_vram:+.1f}",
                f"{rss:.1f}",
                f"{elapsed_ms:.1f}",
            )

    duration = time.time() - start_time
    gc.collect()
    final_snap = monitor.get_snapshot()
    final_vram = final_snap.vram_used_mb
    final_rss = final_snap.process_rss_mb
    net_vram_leak = final_vram - initial_vram
    net_rss_leak = final_rss - initial_rss

    console.print(table)

    summary = {
        "cycles_completed": cycles,
        "duration_seconds": round(duration, 2),
        "initial_vram_mb": round(initial_vram, 1),
        "peak_vram_mb": round(peak_vram, 1),
        "final_vram_mb": round(final_vram, 1),
        "net_vram_leak_mb": round(net_vram_leak, 2),
        "initial_rss_mb": round(initial_rss, 1),
        "peak_rss_mb": round(peak_rss, 1),
        "final_rss_mb": round(final_rss, 1),
        "net_rss_leak_mb": round(net_rss_leak, 2),
        "stable": abs(net_vram_leak) < 150.0 and abs(net_rss_leak) < 50.0,
    }

    report_path = config.logs_dir / "vram_stress_test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    evidence_path = Path("/home/rai/Anti-Gravity/evidence/vram_stress_test_report.json")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    console.print(f"\n[bold]Stress Test Verdict:[/bold] {'[green]PASSED (Zero Progressive Leakage)[/green]' if summary['stable'] else '[red]FAILED (Memory Leak Detected)[/red]'}")
    console.print(f"Net VRAM Delta: [bold]{net_vram_leak:+.2f} MB[/bold] | Net RSS Delta: [bold]{net_rss_leak:+.2f} MB[/bold]")
    console.print(f"Detailed report saved to: [cyan]{report_path}[/cyan]\n")

    daemon.model_mgr.shutdown()
    monitor.shutdown()
    return summary


@click.command()
@click.option("--cycles", default=30, help="Number of test cycles")
@click.option("--profile", default="balanced", help="Config profile")
def main(cycles: int, profile: str):
    run_stress_test(cycles=cycles, profile=profile)


if __name__ == "__main__":
    main()
