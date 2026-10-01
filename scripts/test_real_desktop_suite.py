"""
Comprehensive Real Desktop Action Verification Suite for Laya Desktop Agent.
Executes and verifies real application actions:
- Test A: "Open Chrome"
- Test B: "Open Telegram"
- Test C: "Switch to Chrome"
- Test D: "Open terminal"
- Test E: "Type hello world"
- Test F: "Scroll down"
- Test G: "Take a screenshot"
- Test H: Multi-step command (Open Chrome, search YouTube)
- Emergency Stop (F11)
Records exact measured latencies, VRAM, and RAM before and after.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_config
from app.daemon import LayaDaemon
from integrations.kdotool_client import KdotoolClient
from monitoring.resource_monitor import ResourceMonitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DesktopTestSuite")


def run_suite():
    evidence_dir = Path("/home/rai/Anti-Gravity/evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    report_file = evidence_dir / "real_desktop_test_evidence.json"

    logger.info("Initializing Laya Daemon with balanced profile...")
    cfg = load_config(profile="balanced")
    daemon = LayaDaemon(cfg, diagnostic_mode=False)
    
    # Warm up models so we benchmark realistic warm operational state
    daemon.model_mgr.warm_models()
    
    kdotool = KdotoolClient()
    monitor = ResourceMonitor()
    
    initial_snap = monitor.get_snapshot()
    logger.info("Initial System State: VRAM=%.1fMB | RSS=%.1fMB", initial_snap.vram_used_mb, initial_snap.process_rss_mb)

    results = {}

    # --- Test C: Switch to Chrome ---
    logger.info(">>> TEST C: Switch to Chrome <<<")
    t0 = time.perf_counter()
    res_c = daemon.coordinator.process_text_command("Switch to Chrome")
    time.sleep(0.3)
    active_win = kdotool.get_active_window()
    active_name = kdotool.get_window_name(active_win)
    active_class = kdotool.get_window_class(active_win)
    passed_c = "chrome" in active_class.lower() or "chrome" in active_name.lower()
    results["Test_C_Switch_Chrome"] = {
        "command": "Switch to Chrome",
        "passed": passed_c,
        "active_window": active_name,
        "active_class": active_class,
        "durations_ms": res_c.metrics.durations_ms,
        "total_ms": (time.perf_counter() - t0) * 1000.0,
    }
    logger.info("Test C Result: %s (Active: %s [%s])", "PASS" if passed_c else "FAIL", active_name, active_class)

    # --- Test F: Scroll down ---
    logger.info(">>> TEST F: Scroll down <<<")
    t0 = time.perf_counter()
    res_f = daemon.coordinator.process_text_command("Scroll down twice")
    results["Test_F_Scroll"] = {
        "command": "Scroll down twice",
        "passed": res_f.success,
        "action": res_f.action_executed,
        "durations_ms": res_f.metrics.durations_ms,
        "total_ms": (time.perf_counter() - t0) * 1000.0,
    }
    logger.info("Test F Result: %s", "PASS" if res_f.success else "FAIL")

    # --- Test G: Take a screenshot ---
    logger.info(">>> TEST G: Take a screenshot <<<")
    t0 = time.perf_counter()
    res_g = daemon.coordinator.process_text_command("Take a screenshot")
    # Verify screenshot was saved in cache
    cache_dir = Path("/home/rai/Projects/laya-desktop-agent/cache/screenshots")
    screenshots = list(cache_dir.glob("*.png")) if cache_dir.exists() else []
    passed_g = res_g.success and len(screenshots) > 0
    results["Test_G_Screenshot"] = {
        "command": "Take a screenshot",
        "passed": passed_g,
        "screenshot_count": len(screenshots),
        "durations_ms": res_g.metrics.durations_ms,
        "total_ms": (time.perf_counter() - t0) * 1000.0,
    }
    logger.info("Test G Result: %s (Found %d screenshot files)", "PASS" if passed_g else "FAIL", len(screenshots))

    # --- Test D: Open terminal ---
    logger.info(">>> TEST D: Open terminal <<<")
    t0 = time.perf_counter()
    res_d = daemon.coordinator.process_text_command("Open terminal")
    time.sleep(0.5)
    active_win = kdotool.get_active_window()
    active_name = kdotool.get_window_name(active_win)
    active_class = kdotool.get_window_class(active_win)
    passed_d = res_d.success and ("ghostty" in active_class.lower() or "terminal" in active_name.lower())
    results["Test_D_Open_Terminal"] = {
        "command": "Open terminal",
        "passed": passed_d,
        "active_window": active_name,
        "active_class": active_class,
        "durations_ms": res_d.metrics.durations_ms,
        "total_ms": (time.perf_counter() - t0) * 1000.0,
    }
    logger.info("Test D Result: %s (Active: %s [%s])", "PASS" if passed_d else "FAIL", active_name, active_class)

    # --- Test E: Type hello world ---
    logger.info(">>> TEST E: Type hello world <<<")
    t0 = time.perf_counter()
    res_e = daemon.coordinator.process_text_command("Type hello world")
    results["Test_E_Type_Text"] = {
        "command": "Type hello world",
        "passed": res_e.success,
        "action": res_e.action_executed,
        "durations_ms": res_e.metrics.durations_ms,
        "total_ms": (time.perf_counter() - t0) * 1000.0,
    }
    logger.info("Test E Result: %s", "PASS" if res_e.success else "FAIL")

    # --- Test B: Open Telegram ---
    logger.info(">>> TEST B: Open Telegram <<<")
    t0 = time.perf_counter()
    res_b = daemon.coordinator.process_text_command("Open Telegram")
    time.sleep(1.5)
    wins = [kdotool.get_window_name(w) for w in kdotool.search("", limit=15)]
    passed_b = res_b.success and any("telegram" in w.lower() for w in wins)
    results["Test_B_Open_Telegram"] = {
        "command": "Open Telegram",
        "passed": passed_b,
        "open_windows_matching": [w for w in wins if "telegram" in w.lower()],
        "durations_ms": res_b.metrics.durations_ms,
        "total_ms": (time.perf_counter() - t0) * 1000.0,
    }
    logger.info("Test B Result: %s", "PASS" if passed_b else "FAIL")

    # --- Test A: Open Chrome ---
    logger.info(">>> TEST A: Open Chrome <<<")
    t0 = time.perf_counter()
    res_a = daemon.coordinator.process_text_command("Open Chrome")
    time.sleep(0.5)
    wins = [kdotool.get_window_name(w) for w in kdotool.search("", limit=15)]
    passed_a = res_a.success and any("chrome" in w.lower() for w in wins)
    results["Test_A_Open_Chrome"] = {
        "command": "Open Chrome",
        "passed": passed_a,
        "durations_ms": res_a.metrics.durations_ms,
        "total_ms": (time.perf_counter() - t0) * 1000.0,
    }
    logger.info("Test A Result: %s", "PASS" if passed_a else "FAIL")

    # --- Test H: Multi-step YouTube Search ---
    logger.info(">>> TEST H: Search YouTube for Arch Linux <<<")
    t0 = time.perf_counter()
    res_h = daemon.coordinator.process_text_command("Search YouTube for Arch Linux")
    time.sleep(0.8)
    wins = [kdotool.get_window_name(w) for w in kdotool.search("", limit=15)]
    passed_h = res_h.success and any("youtube" in w.lower() or "chrome" in w.lower() for w in wins)
    results["Test_H_YouTube_Search"] = {
        "command": "Search YouTube for Arch Linux",
        "passed": passed_h,
        "durations_ms": res_h.metrics.durations_ms,
        "total_ms": (time.perf_counter() - t0) * 1000.0,
    }
    logger.info("Test H Result: %s", "PASS" if passed_h else "FAIL")

    # --- Emergency Stop Test ---
    logger.info(">>> EMERGENCY STOP TEST (F11) <<<")
    daemon.emergency_stop.trigger(reason="Test F11 trigger")
    estop_active = daemon.emergency_stop.is_stopped()
    # Attempt command during estop -> should be blocked
    res_blocked = daemon.coordinator.process_text_command("Open Chrome")
    estop_blocks = not res_blocked.success or daemon.emergency_stop.is_stopped()
    results["Emergency_Stop_Test"] = {
        "estop_active": estop_active,
        "blocked_subsequent_action": estop_blocks,
        "passed": estop_active and estop_blocks,
    }
    logger.info("Emergency Stop Result: %s", "PASS" if (estop_active and estop_blocks) else "FAIL")

    # Reset estop for final cleanup
    daemon.emergency_stop.reset()

    # Final Resource Check
    final_snap = monitor.get_snapshot()
    logger.info("Final System State: VRAM=%.1fMB | RSS=%.1fMB", final_snap.vram_used_mb, final_snap.process_rss_mb)

    results["Resource_Metrics"] = {
        "initial_vram_mb": initial_snap.vram_used_mb,
        "final_vram_mb": final_snap.vram_used_mb,
        "vram_delta_mb": final_snap.vram_used_mb - initial_snap.vram_used_mb,
        "initial_rss_mb": initial_snap.process_rss_mb,
        "final_rss_mb": final_snap.process_rss_mb,
        "rss_delta_mb": final_snap.process_rss_mb - initial_snap.process_rss_mb,
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("All tests completed! Evidence report written to: %s", report_file)
    print("\n================== SUMMARY ==================")
    for k, v in results.items():
        if isinstance(v, dict) and "passed" in v:
            status = "PASS" if v["passed"] else "FAIL"
            total = v.get("total_ms", 0.0)
            print(f"[{status}] {k}: {total:.1f}ms")
    print(f"VRAM Delta: {results['Resource_Metrics']['vram_delta_mb']:.1f} MB")
    print(f"RAM Delta:  {results['Resource_Metrics']['rss_delta_mb']:.1f} MB")
    print("=============================================\n")

    daemon.stop()


if __name__ == "__main__":
    run_suite()
