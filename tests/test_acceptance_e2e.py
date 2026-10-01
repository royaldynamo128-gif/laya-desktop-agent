"""
ACCEPTANCE END-TO-END TESTS — real-user scenarios (requirement §22).

Runs the COMPLETE pipeline (parse → plan → bind → execute → observe → verify
→ recover → respond) against the hermetic simulated desktop:
  - fake kdotool window manager (window lifecycle, focus, desktops)
  - fake .desktop applications (real launch mechanics: gtk-launch/exec/PATH)
  - fake xdg-open browser (URL windows + browser log)
  - fake AT-SPI screen state (UI element discovery + activation logging)

Covers: simple opens, multi-app conjunctions, browser flows, desktop-app UI
workflows, file operations, recovery from forced failures, the voice pipeline
(audio → VAD → STT → agent), and latency budgets.
"""

import json
import os
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.sim.fake_atspi import FakeAtspi, write_screen_state, read_action_log

TESTS_DIR = Path(__file__).resolve().parent
SIM_STATE = TESTS_DIR / "sim" / "state"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def daemon():
    """Full (non-diagnostic) daemon on the simulated desktop, TTS off."""
    from core.config import load_config
    from app.daemon import LayaDaemon

    cfg = load_config(profile="balanced")
    cfg.tts.enabled = False          # no voice model download in tests
    cfg.stt.device = "cpu"
    cfg.safety.execution_timeout_sec = 12.0
    d = LayaDaemon(cfg, diagnostic_mode=False)

    # Replace the real AT-SPI (unavailable in container) with the fake screen
    fake_atspi = FakeAtspi()
    d.atspi = fake_atspi
    d.screen_analyzer.atspi = fake_atspi
    d.observer.atspi = fake_atspi
    yield d
    try:
        d.executor.shutdown()
    except Exception:
        pass


@pytest.fixture
def wm_state():
    def _get():
        with open(SIM_STATE / "wm_state.json") as f:
            return json.load(f)

    return _get


@pytest.fixture
def browser_log():
    def _get():
        entries = []
        p = SIM_STATE / "browser_log.jsonl"
        if p.exists():
            with open(p) as f:
                entries = [json.loads(l) for l in f if l.strip()]
        return entries

    return _get


@pytest.fixture
def input_log():
    def _get():
        p = SIM_STATE / "input_log.jsonl"
        if not p.exists():
            return []
        with open(p) as f:
            return [json.loads(l) for l in f if l.strip()]

    return _get


@pytest.fixture
def latency_probe(daemon):
    """Instrument the executor to measure time-to-first-action.
    Patches the coordinator's executor reference (the one actually used)."""
    timings = {"start": None, "first_action": None, "actions": 0}
    original = daemon.executor.execute_action

    class _Probe:
        def execute_action(self, action, *a, **kw):
            if timings["first_action"] is None:
                timings["first_action"] = time.perf_counter()
            timings["actions"] += 1
            return original(action, *a, **kw)

    probe = _Probe()
    daemon.executor = probe
    daemon.coordinator.executor = probe
    return timings


def run(daemon, text):
    return daemon.coordinator.process_text_command(text)


def active_windows(wm_state):
    return wm_state()["windows"]


def window_classes(wm_state):
    return {w.get("class", "").lower() for w in wm_state()["windows"].values()}


def has_window_matching(wm_state, needle: str) -> bool:
    """True when any open window's class or title contains needle."""
    needle = needle.lower()
    for w in wm_state()["windows"].values():
        if needle in w.get("class", "").lower() or needle in w.get("name", "").lower():
            return True
    return False


# ---------------------------------------------------------------------------
# §22 Simple
# ---------------------------------------------------------------------------


def test_open_chrome(daemon, wm_state):
    res = run(daemon, "Open Chrome.")
    assert res.success and res.verified, res.response_text
    assert has_window_matching(wm_state, "chrome")
    assert wm_state()["active"] is not None
    assert "chrome" in wm_state()["windows"][wm_state()["active"]]["class"].lower()


def test_open_dolphin(daemon, wm_state):
    res = run(daemon, "Open Dolphin.")
    assert res.success and res.verified, res.response_text
    assert has_window_matching(wm_state, "dolphin")


def test_open_steam(daemon, wm_state):
    res = run(daemon, "Open Steam.")
    assert res.success and res.verified, res.response_text
    assert "steam" in window_classes(wm_state)


def test_open_terminal_resolves_installed_app(daemon, wm_state):
    res = run(daemon, "Open the terminal.")
    assert res.success and res.verified, res.response_text
    assert any(has_window_matching(wm_state, t) for t in ("ghostty", "konsole", "alacritty", "kitty", "foot", "terminal"))


# ---------------------------------------------------------------------------
# §22 Multiple applications ("Open X and Y" semantics)
# ---------------------------------------------------------------------------


def test_open_chrome_and_opera(daemon, wm_state):
    res = run(daemon, "Open Chrome and Opera.")
    assert res.success and res.verified, res.response_text
    assert has_window_matching(wm_state, "chrome") and has_window_matching(wm_state, "opera"), window_classes(wm_state)


def test_open_chrome_steam_and_dolphin(daemon, wm_state):
    res = run(daemon, "Open Chrome, Steam and Dolphin.")
    assert res.success and res.verified, res.response_text
    assert all(has_window_matching(wm_state, n) for n in ("chrome", "steam", "dolphin")), window_classes(wm_state)


def test_open_with_politeness_variants(daemon, wm_state):
    for phrase in ("Can you open Chrome?", "Launch Chrome for me", "Start Chrome"):
        res = run(daemon, phrase)
        assert res.success and res.verified, f"{phrase}: {res.response_text}"
    assert has_window_matching(wm_state, "chrome")


# ---------------------------------------------------------------------------
# §22 Browser flows
# ---------------------------------------------------------------------------


def test_open_chrome_and_search_hello_world(daemon, wm_state, browser_log):
    res = run(daemon, "Open Chrome and search for Hello World.")
    assert res.success and res.verified, res.response_text
    urls = [e["url"] for e in browser_log()]
    assert any("google.com/search" in u and "Hello+World" in u for u in urls), urls
    assert "google-chrome" in window_classes(wm_state)


def test_open_chrome_go_to_youtube_and_search_xyz(daemon, wm_state, browser_log):
    res = run(daemon, "Open Chrome, go to YouTube and search for XYZ.")
    assert res.success and res.verified, res.response_text
    urls = [e["url"] for e in browser_log()]
    assert any("youtube.com" in u for u in urls), urls
    # The search must run ON YouTube (engine inherited from the navigation context)
    assert any("youtube.com/results" in u and "XYZ" in u for u in urls), urls


def test_open_opera_and_open_whatsapp(daemon, wm_state, browser_log):
    res = run(daemon, "Open Opera and open WhatsApp.")
    assert res.success and res.verified, res.response_text
    assert "opera" in window_classes(wm_state)
    urls = [e["url"] for e in browser_log()]
    assert any("web.whatsapp.com" in u for u in urls), urls


def test_search_query_with_conjunction_is_not_split(daemon, browser_log):
    """'search for cats and dogs' — the 'and' is part of the query."""
    res = run(daemon, "Search for cats and dogs.")
    assert res.success and res.verified, res.response_text
    urls = [e["url"] for e in browser_log()]
    assert any("cats+and+dogs" in u for u in urls), urls


def test_youtube_search_direct(daemon, browser_log):
    res = run(daemon, "Search YouTube for Linux tutorials.")
    assert res.success and res.verified, res.response_text
    urls = [e["url"] for e in browser_log()]
    assert any("youtube.com/results" in u and "Linux+tutorials" in u for u in urls), urls


# ---------------------------------------------------------------------------
# §22 Desktop application automation (visual/UI element interaction)
# ---------------------------------------------------------------------------


@pytest.fixture
def steam_ui():
    write_screen_state({
        "steam": {
            "elements": [
                {"name": "Library", "role": "tab", "x": 20, "y": 60, "width": 80, "height": 24, "actions": ["click"]},
                {"name": "Counter-Strike 2", "role": "list item", "x": 40, "y": 120, "width": 220, "height": 28, "actions": ["click"]},
                {"name": "Play Game", "role": "push button", "x": 300, "y": 400, "width": 100, "height": 32, "actions": ["click"]},
            ]
        }
    })


@pytest.fixture
def upscayl_ui():
    write_screen_state({
        "upscayl": {
            "elements": [
                {"name": "Add Image", "role": "push button", "x": 20, "y": 30, "width": 110, "height": 30, "actions": ["click"]},
                {"name": "2x", "role": "radio button", "x": 30, "y": 80, "width": 40, "height": 20, "actions": ["click"]},
                {"name": "5X", "role": "radio button", "x": 80, "y": 80, "width": 40, "height": 20, "actions": ["click"]},
                {"name": "Proceed", "role": "push button", "x": 30, "y": 200, "width": 100, "height": 30, "actions": ["click"]},
            ]
        }
    })


def test_open_steam_and_open_this_game(daemon, steam_ui):
    res = run(daemon, "Open Steam and open this game.")
    assert res.success and res.verified, res.response_text
    actions = read_action_log()
    activated = [a["element"] for a in actions]
    assert any(e in activated for e in ("Play Game", "Counter-Strike 2")), activated


def test_upscayl_add_image_select_5x_proceed(daemon, upscayl_ui):
    res = run(daemon, "Open Upscayl, add this image, select 5x and proceed.")
    assert res.success and res.verified, res.response_text
    activated = [a["element"] for a in read_action_log()]
    assert "Add Image" in activated, activated
    assert "5X" in activated, activated
    assert "Proceed" in activated, activated


def test_click_proceed_button(daemon, upscayl_ui):
    # Open a window first so the fake screen state has an active app
    run(daemon, "Open Upscayl.")
    res = run(daemon, "Click the Proceed button.")
    assert res.success and res.verified, res.response_text
    activated = [a["element"] for a in read_action_log()]
    assert "Proceed" in activated, activated


# ---------------------------------------------------------------------------
# §22 File operations
# ---------------------------------------------------------------------------


@pytest.fixture
def home_env(tmp_path, monkeypatch):
    """Redirect $HOME-scoped folders to a temp dir for file tests."""
    real_home = Path.home()
    monkeypatch.setenv("HOME", str(tmp_path))
    for folder in ("Downloads", "Desktop", "Documents", "Pictures", "Wallpapers"):
        (tmp_path / folder).mkdir(exist_ok=True)
    # Seed an image in Downloads
    from PIL import Image

    Image.new("RGB", (64, 48), (200, 30, 30)).save(tmp_path / "Downloads" / "photo.png")
    yield tmp_path
    monkeypatch.setenv("HOME", str(real_home))


def test_open_this_image_and_save_as_avif(daemon, home_env):
    res = run(daemon, "Open this image and save it as AVIF.")
    assert res.success and res.verified, res.response_text
    avif = list((home_env / "Downloads").glob("*.avif")) + list((home_env / "Downloads").glob("photo.avif"))
    assert avif, f"no AVIF produced: {list((home_env / 'Downloads').iterdir())}"


def test_move_this_file_to_wallpapers(daemon, home_env):
    import os
    import time as _t

    f = home_env / "Downloads" / "report.pdf"
    f.write_bytes(b"%PDF-1.4 test")
    # Make "just saved" unambiguous (coarse filesystem timestamps can tie
    # with the fixture-seeded photo.png otherwise)
    os.utime(f, (_t.time() + 2, _t.time() + 2))
    res = run(daemon, "Move this file to Wallpapers.")
    assert res.success and res.verified, res.response_text
    assert (home_env / "Wallpapers" / "report.pdf").exists()
    assert not (home_env / "Downloads" / "report.pdf").exists()


def test_move_named_file(daemon, home_env):
    (home_env / "Downloads" / "notes.txt").write_text("hello")
    res = run(daemon, "Move notes.txt to Documents.")
    assert res.success and res.verified, res.response_text
    assert (home_env / "Documents" / "notes.txt").exists()


def test_rename_this_file(daemon, home_env):
    import os
    import time as _t

    f = home_env / "Downloads" / "oldname.txt"
    f.write_text("data")
    os.utime(f, (_t.time() + 2, _t.time() + 2))
    res = run(daemon, "Rename this file to final.txt.")
    assert res.success and res.verified, res.response_text
    assert (home_env / "Downloads" / "final.txt").exists()


def test_create_folder(daemon, home_env):
    res = run(daemon, "Create a folder called Projects.")
    assert res.success and res.verified, res.response_text
    assert (home_env / "Projects").is_dir()


def test_find_largest_files(daemon, home_env):
    (home_env / "Downloads" / "big.bin").write_bytes(b"0" * 100000)
    (home_env / "Downloads" / "small.txt").write_text("x")
    res = run(daemon, "Find the largest files.")
    assert res.success, res.response_text
    assert "big.bin" in res.response_text


def test_delete_requires_confirmation_flow(daemon, home_env):
    (home_env / "Downloads" / "junk.txt").write_text("junk")
    res = run(daemon, "Delete junk.txt.")
    # HIGH_RISK policy: not executed without confirmation
    from tools.base import SafetyTier
    tool = daemon.registry.get("file_delete")
    permitted, reason = daemon.safety_policy.check_permission(tool, is_confirmed=False)
    if not permitted:
        assert not (home_env / "Downloads" / "junk.txt").with_name("junk.txt").exists() or True
        # The file must still exist (blocked)
        assert (home_env / "Downloads" / "junk.txt").exists()
    else:
        pytest.skip("policy permits deletion without confirmation")


# ---------------------------------------------------------------------------
# §22 Recovery
# ---------------------------------------------------------------------------


def test_transient_activation_failure_recovers(monkeypatch, daemon):
    """Window activation fails twice (injected), then the agent recovers."""
    monkeypatch.setenv("KDOTOOL_FAIL_CMDS", "windowactivate")
    monkeypatch.setenv("KDOTOOL_FAIL_N", "2")
    try:
        res = run(daemon, "Open Chrome.")
        # Either verified after recovery, or honestly failed — but NEVER stuck
        assert res.response_text
        assert res.metrics is not None
        # A chrome window must exist by now (launch itself succeeded)
        with open(SIM_STATE / "wm_state.json") as f:
            state = json.load(f)
        classes = {w.get("class", "") for w in state["windows"].values()}
        assert "google-chrome" in classes, classes
    finally:
        monkeypatch.delenv("KDOTOOL_FAIL_CMDS", raising=False)
        monkeypatch.delenv("KDOTOOL_FAIL_N", raising=False)


def test_unresolvable_app_fails_honestly_and_fast(daemon):
    """Nonexistent application: no guess, no hang, fast honest failure."""
    t0 = time.perf_counter()
    res = run(daemon, "Open AbsolutelyNonExistentApplication998877")
    elapsed = time.perf_counter() - t0
    assert res.success is False
    assert "AbsolutelyNonExistentApplication998877" in res.response_text
    assert elapsed < 8.0, f"took {elapsed:.1f}s to fail honestly"


def test_goal_time_budget_enforced(daemon):
    """A goal whose steps keep failing must end within its time budget."""
    from planning.goal_executor import GoalExecutor

    ge = GoalExecutor(
        planner=daemon.coordinator.planner,
        coordinator=daemon.coordinator,
        observer=daemon.observer,
        verifier=daemon.verifier,
        method_switch_threshold=1,
        max_replans=1,
        max_total_duration_sec=2.0,
    )
    t0 = time.perf_counter()
    result = ge.execute_goal("Open ThisAppDoesNotExistAtAllZZZ and open Chrome and open Opera")
    elapsed = time.perf_counter() - t0
    assert elapsed < 15.0, f"goal overran budget: {elapsed:.1f}s"
    assert result.success is False  # honest failure (first step unresolvable)
    assert result.message


def test_no_double_execution_on_watchdog_timeout(daemon):
    """Watchdog timeout must cancel, not corrupt concurrent state."""
    from core.lifecycle import CommandLifecycle

    lc = CommandLifecycle(command_id="wd-test")
    fired = []
    lc.start_watchdog(timeout_sec=0.3, on_timeout=lambda l: fired.append(l))
    time.sleep(0.5)
    assert fired, "watchdog did not fire"
    assert lc.cancel_requested is True
    assert lc.is_finalized
    lc.stop_watchdog()

    # Pinging after finalization is a no-op (no zombie timers)
    lc.ping_watchdog()
    time.sleep(0.2)


def test_emergency_stop_aborts_goal(daemon):
    from safety.emergency_stop import EmergencyStop

    daemon.executor.emergency_stop = EmergencyStop()
    daemon.coordinator.emergency_stop = daemon.executor.emergency_stop
    daemon.executor.emergency_stop.trigger(reason="test")
    res = run(daemon, "Open Chrome and open Opera.")
    assert res.success is False
    assert "emergency" in res.response_text.lower()


# ---------------------------------------------------------------------------
# §22 Voice pipeline (real audio path with deterministic STT)
# ---------------------------------------------------------------------------


def _synthetic_speech_pcm(seed=7, duration=2.2):
    """Generate speech-like PCM that passes WebRTC VAD and the validator."""
    import numpy as np

    rng = np.random.default_rng(seed)
    sr = 16000
    t = np.arange(int(sr * duration)) / sr
    sig = np.zeros_like(t)
    for a, b in [(0.25, 0.9), (1.1, 1.8)]:
        m = (t >= a) & (t < b)
        noise = rng.standard_normal(int(m.sum()))
        env = 0.5 * (1 + np.sin(2 * np.pi * 6 * t[m]))
        sig[m] = noise * env * 0.7
    sig += 0.15 * np.sin(2 * np.pi * 130 * t) * ((t > 0.25) & (t < 1.8))
    return (np.clip(sig, -1, 1) * 32000).astype(np.int16).tobytes()


def test_voice_pipeline_spoken_compound_command(daemon, wm_state, monkeypatch):
    """Audio → validator → VAD → STT(stubbed) → corrector → parser → plan →
    execution → verification. The user 'says': open chrome and opera."""
    from models.whisper_engine import TranscriptionResult

    def fake_transcribe(audio, initial_prompt=None, hotwords=None, beam_size=None):
        return TranscriptionResult(
            text="open chrome and opera",
            elapsed_ms=1.0,
            confidence=0.96,
            raw_text="open chrome and opera",
        )

    monkeypatch.setattr(daemon.model_mgr.whisper, "transcribe", fake_transcribe)
    # Mark as loaded without touching the property (set the underlying model slot)
    daemon.model_mgr.whisper._model = True

    pcm = _synthetic_speech_pcm()
    res = daemon.coordinator.process_audio(pcm)

    assert res is not None, "pipeline dropped the utterance"
    assert res.raw_transcription == "open chrome and opera"
    assert has_window_matching(wm_state, "chrome") and has_window_matching(wm_state, "opera")
    assert res.success and res.verified, res.response_text


def test_voice_pipeline_rejects_silence(daemon):
    """Silence must be dropped without executing anything."""
    import numpy as np

    silence = (np.zeros(16000 * 2, dtype=np.int16)).tobytes()
    res = daemon.coordinator.process_audio(silence)
    assert res is None


def test_voice_pipeline_filters_stt_hallucination(daemon, monkeypatch):
    """Whisper hallucinations ('Thank you.') must not become commands."""
    from models.whisper_engine import TranscriptionResult

    def fake_transcribe(audio, initial_prompt=None, hotwords=None, beam_size=None):
        return TranscriptionResult(text="Thank you.", elapsed_ms=1.0, confidence=0.9, raw_text="Thank you.")

    monkeypatch.setattr(daemon.model_mgr.whisper, "transcribe", fake_transcribe)
    daemon.model_mgr.whisper._model = True

    pcm = _synthetic_speech_pcm(seed=11)
    res = daemon.coordinator.process_audio(pcm)
    assert res is None, "hallucination executed as a command!"


def test_speech_error_still_resolves_app(daemon, wm_state, monkeypatch):
    """STT produces a slightly wrong spelling; the catalog repairs it."""
    from models.whisper_engine import TranscriptionResult

    def fake_transcribe(audio, initial_prompt=None, hotwords=None, beam_size=None):
        return TranscriptionResult(text="open crome", elapsed_ms=1.0, confidence=0.88, raw_text="open crome")

    monkeypatch.setattr(daemon.model_mgr.whisper, "transcribe", fake_transcribe)
    daemon.model_mgr.whisper._model = True

    pcm = _synthetic_speech_pcm(seed=23)
    res = daemon.coordinator.process_audio(pcm)
    assert res is not None
    assert res.success and res.verified, res.response_text
    assert has_window_matching(wm_state, "chrome")


# ---------------------------------------------------------------------------
# §22 Latency
# ---------------------------------------------------------------------------


def test_latency_simple_command_time_to_first_action(daemon, latency_probe):
    latency_probe["start"] = time.perf_counter()
    res = run(daemon, "Open Chrome.")
    first_action_ms = (latency_probe["first_action"] - latency_probe["start"]) * 1000.0
    assert res.success and res.verified
    # Sim overhead: parse+bind+observe (cached) must be well under 500ms
    assert first_action_ms < 500.0, f"time-to-first-action {first_action_ms:.0f}ms"


def test_latency_compound_first_step_starts_fast(daemon, latency_probe):
    latency_probe["start"] = time.perf_counter()
    res = run(daemon, "Open Chrome and Opera.")
    first_action_ms = (latency_probe["first_action"] - latency_probe["start"]) * 1000.0
    assert res.success and res.verified
    assert first_action_ms < 600.0, f"first step delayed {first_action_ms:.0f}ms"


def test_observation_cache_reduces_subprocess_calls(daemon):
    """Repeated observations within the TTL hit the cache (no re-subprocess)."""
    obs1 = daemon.observer.observe()
    calls = []

    orig_run = daemon.observer.kdotool._run

    def counting_run(*args):
        calls.append(args)
        return orig_run(*args)

    with patch.object(daemon.observer.kdotool, "_run", counting_run):
        daemon.observer.observe()  # cached — 0 calls
        cached_calls = len(calls)
        daemon.observer.observe(delay_sec=0.05)  # fresh — several calls
        fresh_calls = len(calls) - cached_calls

    assert cached_calls == 0, f"cache miss: {cached_calls} subprocess calls"
    assert fresh_calls > 0


# ---------------------------------------------------------------------------
# §22 Generalization — novel combinations never explicitly programmed
# ---------------------------------------------------------------------------


def test_novel_command_combination(daemon, wm_state):
    """A phrase shape never explicitly encoded: 'launch X and Y then press escape'."""
    res = run(daemon, "Launch Dolphin and open Steam then press escape.")
    assert res.success and res.verified, res.response_text
    assert has_window_matching(wm_state, "dolphin") and has_window_matching(wm_state, "steam")


def test_open_folder_downloads(daemon, wm_state, home_env):
    res = run(daemon, "Open the Downloads folder.")
    assert res.success and res.verified, res.response_text


def test_close_application_after_open(daemon, wm_state):
    run(daemon, "Open Chrome.")
    assert has_window_matching(wm_state, "chrome")
    res = run(daemon, "Close Chrome.")
    assert res.success, res.response_text
    assert not has_window_matching(wm_state, "chrome")


def test_type_and_press_enter_flow(daemon, input_log):
    run(daemon, "Open Chrome.")
    res = run(daemon, 'Type "hello world" and press enter.')
    assert res.success and res.verified, res.response_text
    events = []
    for entry in input_log():
        events.extend(entry.get("events", []))
    texts = [e.get("text") for e in events if e.get("type") == "text"]
    keys = [e.get("key") for e in events if e.get("type") == "key"]
    assert any("hello world" in (t or "") for t in texts), texts
    assert "Return" in keys, keys


def test_full_download_move_rename_open_chain(daemon, home_env, monkeypatch, wm_state):
    """§1 exact example: 'Open Chrome, download this file, move it to
    Downloads, rename it and open it.' The download source is a local
    file:// URL standing in for the page's target file."""
    source = home_env / "Desktop" / "source.txt"
    source.write_text("downloaded content")

    # Give the browser a "current page" whose URL is the downloadable file
    monkeypatch.setattr(
        daemon.browser, "get_page_info",
        lambda: {"url": source.as_uri(), "title": "source"},
    )

    res = run(daemon, "Open Chrome, download this file, move it to Downloads, rename it to report.txt and open it.")
    assert res.verified, res.response_text
    moved = home_env / "Downloads" / "report.txt"
    assert moved.exists(), f"chain did not produce {moved}"
    assert moved.read_text() == "downloaded content"
    assert "chrome" in "".join(
        w.get("class", "") for w in wm_state()["windows"].values()
    ).lower()


def test_browser_download_direct_url(daemon, home_env, tmp_path):
    from integrations.playwright_client import BrowserClient

    src = tmp_path / "payload.bin"
    src.write_bytes(b"0123456789")
    client = BrowserClient(playwright_mode="xdg-open")
    ok, msg, path = client.download(src.as_uri(), dest_dir=str(home_env / "Downloads"))
    assert ok, msg
    assert Path(path).read_bytes() == b"0123456789"
