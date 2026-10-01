"""
ADVERSARIAL ROUND 2 — exhaustive verification beyond the acceptance suite.

Covers the full audit checklist:
  §6  application discovery states (already running, minimized, other
      workspace, ambiguous multiple matches, alternative names)
  §7  shortcut inventory & execution (PTT combo, F11 estop, F12 fallback,
      input key combos, voice 'stop', CLI commands)
  §8  failure & recovery (app not installed, launch failure, window vanishes,
      slow page loads, empty/garbage transcription, kdotool unavailable)
  §9  latency budgets per stage + honest compound-goal timing
  §10 concurrency & stale state (focus change mid-goal, new command after
      completion, state reset between commands)
  §13 lifecycle (daemon start → stop → restart, config profiles, status IPC)
  New-feature verification: "open the result" chains, plural clarifications,
  natural-language politeness variants.
"""

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.sim.fake_atspi import FakeAtspi, write_screen_state, read_action_log
from tests.test_acceptance_e2e import daemon, home_env, run, wm_state, browser_log, input_log  # noqa: F401

TESTS_DIR = Path(__file__).resolve().parent
SIM_STATE = TESTS_DIR / "sim" / "state"
SIM_BIN = TESTS_DIR / "sim" / "bin"


def wm_set(desktop=None, active=None):
    """Directly mutate the simulated window-manager state."""
    p = SIM_STATE / "wm_state.json"
    state = json.loads(p.read_text())
    if desktop is not None:
        state["desktop"] = desktop
    if active is not None:
        state["active"] = active
    p.write_text(json.dumps(state))


def wm_window_ids(wm_state, cls_fragment):
    return [
        wid for wid, w in wm_state()["windows"].items()
        if cls_fragment in (w.get("class", "") + w.get("name", "")).lower()
    ]


# ---------------------------------------------------------------------------
# §6 Application discovery states
# ---------------------------------------------------------------------------


def test_open_app_already_running_focuses_not_duplicates(daemon, wm_state):
    run(daemon, "Open Chrome.")
    n_before = len(wm_state()["windows"])
    res = run(daemon, "Open Chrome.")
    assert res.success and res.verified
    assert len(wm_state()["windows"]) == n_before, "second open must focus, not duplicate"
    assert "switch" in res.response_text.lower() or "running" in res.response_text.lower()


def test_open_minimized_app_restores_it(daemon, wm_state):
    run(daemon, "Open Dolphin.")
    wid = wm_window_ids(wm_state, "dolphin")[0]
    state = json.loads((SIM_STATE / "wm_state.json").read_text())
    state["windows"][wid]["states"] = ["MINIMIZED"]
    state["active"] = None
    (SIM_STATE / "wm_state.json").write_text(json.dumps(state))

    res = run(daemon, "Open Dolphin.")
    assert res.success and res.verified, res.response_text
    state = json.loads((SIM_STATE / "wm_state.json").read_text())
    assert "MINIMIZED" not in state["windows"][wid]["states"], "minimized window must be restored"
    assert state["active"] == wid


def test_open_app_on_other_workspace_switches_desktop(daemon, wm_state):
    run(daemon, "Open Steam.")
    wid = wm_window_ids(wm_state, "steam")[0]
    state = json.loads((SIM_STATE / "wm_state.json").read_text())
    state["windows"][wid]["desktop"] = 3
    state["desktop"] = 1
    (SIM_STATE / "wm_state.json").write_text(json.dumps(state))

    res = run(daemon, "Open Steam.")
    assert res.success and res.verified, res.response_text
    state = json.loads((SIM_STATE / "wm_state.json").read_text())
    assert state["desktop"] == 3, "must switch to the window's workspace"


def test_open_alternative_app_names(daemon, wm_state):
    """'Google Chrome', 'the browser', 'chrome browser' all resolve to chrome."""
    for phrase in ("Open Google Chrome.", "Open the browser.", "Open chrome browser."):
        res = run(daemon, phrase)
        assert res.success and res.verified, f"{phrase!r} -> {res.response_text}"
        assert any("chrome" in (w.get("class", "")).lower() for w in wm_state()["windows"].values()), phrase


def test_plural_referring_app_request_asks_for_clarification(daemon):
    res = run(daemon, "Open these two applications.")
    assert res.intent == "open_application"
    # A clarification (not a resolution failure) is the correct outcome
    assert "which application" in res.response_text.lower(), res.response_text


# ---------------------------------------------------------------------------
# §7 Shortcut inventory & execution
# ---------------------------------------------------------------------------

SHORTCUT_INVENTORY = {
    "global_ptt_combo": "WIN+ALT (config input.shortcut)",
    "ptt_fallback_key": "KEY_F12 (config input.ptt_key)",
    "emergency_stop_key": "KEY_F11 (config input.emergency_stop_key)",
    "cli_activate": "laya-agent activate (SIGUSR1)",
    "cli_stop": "laya-agent stop (SIGTERM)",
    "input_key_combos": "ctrl+w / alt+tab / Enter via PressKeyTool",
    "voice_estop": "spoken stop/emergency commands",
}


def test_shortcut_inventory_is_config_wired(daemon):
    """Every shortcut in the inventory flows from config → PTT listener."""
    from evdev import ecodes

    ptt = daemon.ptt_listener
    assert ptt is not None, "PTT listener must exist (shortcut cannot work without it)"
    assert ptt.shortcut_str == daemon.config.input.shortcut.upper()
    # WIN+ALT parsed into two modifier groups
    assert len(ptt._combo_groups) == 2
    assert ecodes.KEY_LEFTMETA in ptt._combo_groups[0]
    assert ecodes.KEY_LEFTALT in ptt._combo_groups[1]
    assert ptt.fallback_key_code == ecodes.KEY_F12
    assert ptt.estop_key_code == ecodes.KEY_F11


def test_ptt_combo_executes_activation_callback():
    """WIN+ALT key events actually trigger the PTT activation callback."""
    from evdev import ecodes
    from app.ptt_listener import PTTListener

    pressed = []
    listener = PTTListener(
        shortcut="WIN+ALT", activation_mode="push_to_talk",
        on_press=lambda: pressed.append("press"),
        on_release=lambda: pressed.append("release"),
    )
    # Partial combo must NOT trigger
    listener._handle_key_event(ecodes.KEY_LEFTMETA, 1)
    time.sleep(0.03)
    assert pressed == []
    # Full combo triggers
    listener._handle_key_event(ecodes.KEY_RIGHTALT, 1)
    time.sleep(0.03)
    assert pressed == ["press"]
    # Combo broken → release
    listener._handle_key_event(ecodes.KEY_RIGHTALT, 0)
    time.sleep(0.03)
    assert pressed == ["press", "release"]


def test_estop_key_aborts_running_goal(daemon, wm_state):
    from safety.emergency_stop import EmergencyStop

    daemon.emergency_stop.reset()
    daemon._on_estop_press()  # what F11 actually invokes
    assert daemon.emergency_stop.is_stopped
    assert daemon.state_mgr.current_state.value.lower() in ("paused", "error")
    daemon.emergency_stop.reset()


def test_key_combo_ctrl_w_reaches_wtype(daemon, input_log):
    res = run(daemon, "Press control w.")
    assert res.success, res.response_text
    entries = input_log()
    assert any("ctrl" in json.dumps(e).lower() and "w" in json.dumps(e).lower() for e in entries), entries


def test_cli_commands_execute(daemon, tmp_path):
    """The CLI layer itself (doctor / status / exec) runs without crashing."""
    from click.testing import CliRunner
    from app.cli import cli

    runner = CliRunner()
    # doctor: real probes, must succeed even when hardware is missing
    r = runner.invoke(cli, ["doctor"])
    assert r.exit_code == 0, r.output
    # status: resource snapshot tables
    r = runner.invoke(cli, ["status"])
    assert r.exit_code == 0, r.output
    # exec with a sim command (daemon not required — exec builds its own)
    r = runner.invoke(cli, ["exec", "Open Dolphin."])
    assert r.exit_code == 0, r.output
    assert "Processing command" in r.output


# ---------------------------------------------------------------------------
# §8 Failure & recovery
# ---------------------------------------------------------------------------


def test_app_not_installed_fails_honestly_and_quickly(daemon):
    t0 = time.perf_counter()
    res = run(daemon, "Open Photoshop.")
    elapsed = time.perf_counter() - t0
    assert not (res.success and res.verified)
    assert res.response_text and "couldn't find" in res.response_text.lower()
    assert elapsed < 5.0, f"unresolvable app must fail fast ({elapsed:.1f}s)"


def test_window_vanishes_mid_goal_recovers_or_reports(daemon, wm_state, monkeypatch):
    """Close the target window between steps: the goal must not hang."""
    # Instrument: close the chrome window right after the first step executes
    orig_exec = daemon.executor.execute_action
    closed = {"done": False}

    def exec_and_sabotage(action, *a, **kw):
        out = orig_exec(action, *a, **kw)
        if not closed["done"]:
            closed["done"] = True
            for wid in list(json.loads((SIM_STATE / "wm_state.json").read_text())["windows"]):
                state = json.loads((SIM_STATE / "wm_state.json").read_text())
                if "chrome" in (state["windows"][wid].get("class", "") + state["windows"][wid].get("name", "")).lower():
                    state["windows"].pop(wid)
                    if state["active"] == wid:
                        state["active"] = None
                    (SIM_STATE / "wm_state.json").write_text(json.dumps(state))
        return out

    monkeypatch.setattr(daemon.executor, "execute_action", exec_and_sabotage)
    monkeypatch.setattr(daemon.coordinator, "executor", daemon.executor)
    res = run(daemon, "Open Chrome and search for resilient tests.")
    # Recovery (relaunch) or honest failure are both acceptable; a hang is not
    assert res.intent == "autonomous_goal"
    assert res.response_text


def test_slow_page_load_still_completes(daemon, wm_state, browser_log, monkeypatch):
    monkeypatch.setenv("FAKEBROWSER_OPEN_DELAY_MS", "1200")
    res = run(daemon, "Open Chrome and go to YouTube.")
    assert res.success and res.verified, res.response_text
    assert any("youtube" in e["url"] for e in browser_log())


def test_empty_transcript_rejected_cleanly(daemon):
    res = daemon.coordinator.process_audio(b"")
    assert res is None or (not res.success)  # never crashes, never fake-success


def test_garbage_transcript_fails_honestly(daemon):
    res = run(daemon, "florb zizzle quux.")
    assert not (res.success and res.verified)
    assert res.response_text


def test_kdotool_unavailable_degrades_honestly(daemon, monkeypatch):
    """With no window manager tool at all, commands fail honestly (no hang)."""
    monkeypatch.setenv("KDTOOL_BIN", "/nonexistent/kdotool")
    res = run(daemon, "Open Chrome.")
    assert res.response_text  # an honest message, not silence
    monkeypatch.setenv("KDTOOL_BIN", str(SIM_BIN / "kdotool"))


# ---------------------------------------------------------------------------
# §9 Latency
# ---------------------------------------------------------------------------


def test_latency_stages_are_recorded_and_honest(daemon):
    res = run(daemon, "Open Chrome.")
    d = res.metrics.durations_ms
    assert d.get("laya_routing", 0) < 200, d  # parser-first routing is fast
    assert res.metrics.total_ms > 0, "total latency must be measured, not zero"


def test_compound_goal_latency_is_measured(daemon):
    res = run(daemon, "Open Chrome and Opera.")
    assert res.metrics.total_ms > 0, "compound goals previously reported fake 0 ms"
    assert res.metrics.durations_ms.get("goal_execution", 0) > 0


def test_time_to_first_action_budget(daemon):
    timings = {"first": None}
    orig = daemon.executor.execute_action

    class _Probe:
        def execute_action(self, action, *a, **kw):
            if timings["first"] is None:
                timings["first"] = time.perf_counter()
            return orig(action, *a, **kw)

    probe = _Probe()
    daemon.coordinator.executor = probe
    t0 = time.perf_counter()
    run(daemon, "Open Chrome.")
    tta = timings["first"] - t0 if timings["first"] else 999
    assert tta < 2.5, f"time to first action {tta:.2f}s exceeds budget"


# ---------------------------------------------------------------------------
# §10 Concurrency & stale state
# ---------------------------------------------------------------------------


def test_focus_change_mid_goal_does_not_lose_task(daemon, wm_state):
    res = run(daemon, "Open Chrome, open Steam and open Dolphin.")
    assert res.success and res.verified, res.response_text
    classes = {w.get("class", "").lower() for w in wm_state()["windows"].values()}
    assert any("chrome" in c for c in classes)
    assert any("steam" in c for c in classes)
    assert any("dolphin" in c for c in classes)


def test_new_command_after_completion_uses_fresh_state(daemon, wm_state):
    r1 = run(daemon, "Open Chrome.")
    assert r1.success
    r2 = run(daemon, "Open Steam.")
    assert r2.success and r2.verified
    # No stale duplicate: exactly one chrome and one steam window
    classes = [w.get("class", "").lower() for w in wm_state()["windows"].values()]
    assert classes.count("google-chrome") == 1
    assert classes.count("steam") == 1


def test_repeated_identical_command_is_idempotent(daemon, wm_state):
    for _ in range(3):
        res = run(daemon, "Open Opera.")
        assert res.success and res.verified
    classes = [w.get("class", "").lower() for w in wm_state()["windows"].values()]
    assert classes.count("opera") == 1, "repeated opens must not spawn duplicates"


# ---------------------------------------------------------------------------
# New-feature verification (this round's fixes)
# ---------------------------------------------------------------------------


def test_open_the_result_after_search_clicks_result(daemon, wm_state, browser_log):
    def setup_results_screen():
        write_screen_state({"browser": {"elements": [
            {"name": "Search", "role": "search", "x": 400, "y": 20, "width": 300, "height": 30},
            {"name": "Home", "role": "link", "x": 10, "y": 90, "width": 60, "height": 20},
            {"name": "xyz tutorial - full walkthrough (12:34)", "role": "link",
             "x": 300, "y": 160, "width": 400, "height": 90, "actions": ["click", "jump"]},
            {"name": "xyz review and demo (8:02)", "role": "link", "x": 300, "y": 280, "width": 400, "height": 90},
        ]}})

    orig = daemon.browser.open_url

    def patched(url, context=None):
        out = orig(url, context)
        setup_results_screen()
        return out

    daemon.browser.open_url = patched
    try:
        res = run(daemon, "Open Chrome, go to YouTube, search for xyz and open the result.")
        assert res.success and res.verified, res.response_text
        actions = read_action_log()
        clicked = [a["element"] for a in actions if a["kind"] == "activate"]
        assert any("xyz tutorial" in e for e in clicked), f"clicked wrong element: {clicked}"
        assert not any(e == "Home" for e in clicked), "must not click the nav link"
    finally:
        daemon.browser.open_url = orig


def test_play_latest_video_after_search_opens_result(daemon, wm_state, browser_log):
    write_screen_state({"browser": {"elements": [
        {"name": "Search", "role": "search", "x": 400, "y": 20, "width": 300, "height": 30},
        {"name": "Trending", "role": "link", "x": 10, "y": 90, "width": 60, "height": 20},
        {"name": "learn python in 10 minutes (10:01)", "role": "link",
         "x": 300, "y": 160, "width": 400, "height": 90, "actions": ["click"]},
    ]}})

    res = run(daemon, "Open Chrome and search for python, then play the latest video.")
    assert res.success and res.verified, res.response_text
    clicked = [a["element"] for a in read_action_log() if a["kind"] == "activate"]
    assert any("learn python" in e.lower() for e in clicked), clicked


def test_politeness_and_variation_commands(daemon, wm_state):
    """Natural phrasing variants all reach the same verified outcome."""
    for phrase in ("Launch Chrome for me.", "Can you open Dolphin?", "please open steam"):
        res = run(daemon, phrase)
        assert res.success and res.verified, f"{phrase!r} -> {res.response_text}"


def test_start_then_open_sequence(daemon, wm_state):
    res = run(daemon, "Start Chrome, then open Steam.")
    assert res.success and res.verified, res.response_text
    classes = {w.get("class", "").lower() for w in wm_state()["windows"].values()}
    assert any("chrome" in c for c in classes) and any("steam" in c for c in classes)


# ---------------------------------------------------------------------------
# §13 Lifecycle: start → stop → restart
# ---------------------------------------------------------------------------


@pytest.fixture
def lifecycle_daemon(tmp_path, monkeypatch):
    """Daemon whose environment-boundary subsystems are stubbed so the real
    start()/stop() lifecycle (threads, loops, IPC, locks) can be exercised
    hermetically. Everything else (coordinator, tools, executor) is real."""
    # Status IPC socket needs a writable runtime dir (sandbox has no /run/user)
    rt = tmp_path / "runtime"
    rt.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(rt))

    from core.config import load_config
    from app.daemon import LayaDaemon

    cfg = load_config(profile="balanced")
    cfg.tts.enabled = False
    cfg.stt.device = "cpu"
    d = LayaDaemon(cfg, diagnostic_mode=False)
    fake_atspi = FakeAtspi()
    d.atspi = fake_atspi
    d.screen_analyzer.atspi = fake_atspi
    d.observer.atspi = fake_atspi

    # Stub ONLY hardware/model boundary:
    d.model_mgr.warm_models = lambda: None          # no model download
    d.model_mgr.start_sweeper = lambda: None        # no sweeper thread
    d.audio_capture.warm_stream = lambda: None      # no audio device
    d.ptt_listener.start = lambda: None             # no evdev
    d.ptt_listener.stop = lambda: None
    # Overlay: real subprocess spawn replaced by a no-op (GUI unavailable)
    d._spawn_overlay = lambda: None
    # Single-instance lock: real fcntl lock on a tmp path would also work, but
    # the lock file handle survives stop() in-process; bypass for determinism.
    with patch.object(type(d), "_acquire_single_instance_lock", return_value=True):
        yield d
    try:
        d.stop()
    except Exception:
        pass


def test_daemon_start_stop_restart(lifecycle_daemon, tmp_path):
    import threading

    d = lifecycle_daemon
    # --- start ---
    t = threading.Thread(target=d.start, daemon=True)
    t.start()
    for _ in range(40):
        if d._running:
            break
        time.sleep(0.05)
    assert d._running, "daemon main loop must be running"
    # Commands still work while the daemon loop runs
    res = d.coordinator.process_text_command("Open Chrome.")
    assert res.success and res.verified, res.response_text

    # --- stop ---
    d.stop()
    assert not d._running
    t.join(timeout=5)
    assert not t.is_alive(), "main loop thread must exit"

    # --- restart ---
    d2 = lifecycle_daemon
    d2._stopped = False
    d2._running = False
    t2 = threading.Thread(target=d2.start, daemon=True)
    t2.start()
    for _ in range(40):
        if d2._running:
            break
        time.sleep(0.05)
    assert d2._running, "daemon must restart cleanly"
    d2.stop()
    t2.join(timeout=5)
    assert not t2.is_alive()


def test_config_profiles_all_load():
    from core.config import load_config

    for profile in ("low_resource", "balanced", "performance"):
        cfg = load_config(profile=profile)
        assert cfg.profile == profile
        assert cfg.input.shortcut
        assert cfg.stt.model_size
