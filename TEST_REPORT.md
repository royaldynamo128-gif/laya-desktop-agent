# Laya Desktop Agent — Final Test Report

**Date:** 2026-10-02 · **Result:** PASS · **155 tests collected → 150 passed, 5 skipped (hardware-only), 0 failed**

Machine-readable version: `TEST_REPORT.json`

---

## 1. What was tested

Full-pipeline verification on a hermetic simulated desktop (fake kdotool KWin WM with window
lifecycle / focus / desktops / minimize / failure injection, 18 fake `.desktop` apps, fake
`wtype` / `xdg-open` / `gtk-launch`, FakeAtspi accessibility screen with activation logging):

| Area | Verified |
|---|---|
| Startup / shutdown / restart | real `start()` main loop, commands run during the loop, clean stop + restart, 3 config profiles |
| Voice pipeline | audio → VAD → STT → normalization → hallucination filter → routing → execution; silence rejection; STT-error repair |
| Natural-language understanding | politeness variants, 2- and 3-app conjunctions, quoted-query protection, "Start X, then open Y", alternative app names, category fallbacks, novel phrases |
| Multi-step execution | **"Open Chrome, go to YouTube, search for XYZ and open the result"** → 4 verified steps ending in the correct result click; download → rename → move → open chains |
| Application discovery | already-running (focus, no duplicate), minimized (restore), other workspace (switch), ambiguous (clarify), missing (honest fast failure), fuzzy matching |
| Desktop control | keyboard/mouse synthesis, key combos (ctrl+w), typing, scrolling, screenshots |
| Window management | close / minimize / maximize / switch verified against window state |
| Vision / accessibility | element discovery by label and by role (link), OCR fallback path, activation logging |
| Browser automation | navigate/search (engine inheritance), URL normalization, result clicking (DOM + AT-SPI tiers), downloads |
| Filesystem | open/move/copy/rename/delete(confirm)/create-folder/compress/extract/find-largest |
| Error recovery | transient WM failures, window vanishing mid-goal, slow pages, unresolvable apps, garbage/empty transcription, WM tool missing — recovery or honest failure, never a hang |
| Timeouts / cancellation | adaptive watchdog, goal time budget, emergency stop, no double-execution |
| State management | per-goal memory, no stale state between commands, idempotent repeats |
| Metrics | per-stage durations recorded; honest compound-goal totals; resource snapshot |
| CLI | doctor (real probes), status, exec — in-process and as a real subprocess |

## 2. Shortcuts tested

- **WIN+ALT** global PTT combo — config-wired, full key-event simulation (press/release, partial-combo rejection, right-variant keys, hybrid tap/hold)
- **F12** PTT fallback key — parsed and armed
- **F11** emergency stop — aborts a running goal, PAUSED state, notification
- **CLI `activate` / `stop`** — SIGUSR1 / SIGTERM wiring
- **Key combos to apps** (ctrl+w) — verified in the input synthesis log

## 3. Bugs found and fixed this round (11)

1. **Critical packaging:** `planning*` missing from `pyproject.toml` — an installed build would crash at startup (ImportError). Fixed.
2. **Critical functionality:** "open the result" after a search was treated as an app named "result" and failed after garbage rescue replans. Fixed: parser result-frames, goal-executor search-context resolution ("open it", "the video", "play the latest video"), DOM + AT-SPI result-click tiers, engine-aware selectors.
3. **Major:** the result click picked the nav link "Home". Fixed: candidates ranked by query-token overlap + position below the search field.
4. **Major:** compound goals reported `total_ms = 0` (fake latency). Fixed: `goal_execution` measurement + resource snapshot.
5. **Moderate:** "open these two applications" failed silently. Fixed: asks "Which applications would you like me to open?"
6. **Moderate (flaky):** "move this file" could move the wrong file when mtimes tied. Fixed: deterministic (mtime_ns, ctime_ns, path) ordering.
7. **Moderate:** CLI `doctor` showed fake READY states. Fixed: real probes for mic/VAD/STT/kdotool/overlay.
8. **Minor:** doctor used default instead of configured PTT values. Fixed.
9. **Minor:** dead empty `speech/` package shipped. Removed.
10. **Refactor:** overlay spawn extracted to `_spawn_overlay()` (testable, deduplicated).
11. **Minor:** missing `shutil` import in `cli.py` crashed doctor. Fixed; verified by real subprocess run.

Plus **38 architectural defects** repaired in the prior round (double-execution race, watchdog
deadlock, fake verifiers, fake method-switching, dead GoalVerifier, missing file/vision/browser
capabilities, and more — see `worklog.md`).

## 4. Regression results

- Baseline at session start: 122 passed / 5 skipped
- Final run: **150 passed, 5 skipped, 0 failed** (2 min) — +28 new adversarial tests, no regression.

## 5. Remaining external limitations

- 5 evdev tests need a physical keyboard (skip cleanly; covered by simulated key events)
- First-run Whisper download needs network once
- "latest video" = top-relevance result (date sorting would need the site's filter UI)
- Search-result clicking needs Playwright or an accessibility-enabled browser; otherwise honest failure
- Neural router degrades to parser-only fallback when the Convai model is absent (by design)

## 6. How to run

```bash
pip install -r requirements.txt   # or: pip install -e .
pytest tests/                     # full suite (hermetic, no hardware needed)
python -m app.cli exec "Open Chrome and Opera."
python -m app.cli run             # daemon (needs real desktop + mic)
python -m app.cli doctor          # honest subsystem probes
```
