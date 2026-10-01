# Laya Voice Desktop Agent

> **Production-quality, fast, privacy-preserving local voice agent for Arch Linux + KDE Plasma 6 (Wayland) powered by Convai Laya (ModernBERT-large 421M non-autoregressive decision engine) with strict VRAM protection.**

---

## 1. Overview

The **Laya Desktop Agent** allows natural voice control of your Arch Linux desktop with zero cloud dependency. Every component—speech recognition, decision routing, safety gating, OS execution, state verification, and text-to-speech—runs 100% locally on your machine.

### Key Pipeline
$$\text{Microphone} \longrightarrow \text{WebRTC VAD} \longrightarrow \text{Faster-Whisper STT} \longrightarrow \text{Semantic Parser} \longrightarrow \text{Goal Planner} \longrightarrow \text{Goal Executor} \overbrace{\longrightarrow}^{\text{per step: observe \to act \to verify \to recover}} \text{Verifier / GoalVerifier} \longrightarrow \text{Piper TTS}$$

**Natural language is understood by a generalized semantic parser (verb frames + coordination analysis), not a command dictionary.** "Open Chrome and Opera" opens two applications; "search for cats and dogs" is one query; "Open Chrome, go to YouTube and search XYZ" inherits the YouTube search engine from the navigation context. The Laya neural router remains available as a fallback/confidence layer and degrades gracefully when the model is absent.

**Search-result chaining works end-to-end**: "Open Chrome, go to YouTube, search for XYZ and open the result" launches the browser, navigates, searches, and clicks the actual top result (Playwright DOM selectors first; on-screen link-role accessibility elements as fallback — never hardcoded coordinates). Right after a search, "open it", "open the video" and "play the latest video" resolve to the search results through execution context, and underspecified requests such as "open these two applications" produce a clarification question instead of a silent failure.

* **Sub-35ms Decision Routing**: Uses Convai Laya (`laya.Router`), a non-autoregressive "System 1" model that evaluates intent, safety tiers, and observation requirements in a single forward pass (~32.89 ms hot-path inference on CUDA).
* **Strict VRAM Guardrails**: Built for a 6 GB VRAM budget (RTX 3050). The agent never exceeds ~1.3–1.9 GB of VRAM, preserving >4.2 GB of free headroom for your desktop, browser, and games.
* **Wayland & KDE Plasma 6 Native**: Uses `/usr/local/bin/kdotool` for window manipulation, AT-SPI2 (`PyGObject`) for UI accessibility inspection, FreeDesktop/KDE DBus for notifications and MPRIS media playback, and `gtk-launch` for native/flatpak applications.
* **Universal Global Shortcut (`Win + Alt`)**: Global hardware key capture via Linux `evdev` (`KEY_LEFTMETA`/`KEY_RIGHTMETA` + `KEY_LEFTALT`/`KEY_RIGHTALT`), completely bypassing Wayland key-grab restrictions with zero input latency across all applications. Supports **Hybrid Mode** (Hold >350ms = Push-to-Talk, Tap <350ms = Toggle).
* **Instant Emergency Stop**: Hardware-latched global abort key (`KEY_F11` / `Ctrl+Alt+Escape`) that immediately cancels tools, drops action queues, and mutes speech.
* **Zero Host Pollution**: Entire project, virtualenv, models, logs, and caches reside in this single directory (`~/Projects/laya-desktop-agent/`).

---

## 2. System Environment & Requirements

* **Operating System**: Arch Linux (Kernel 6.18+)
* **Desktop Environment**: KDE Plasma 6 on Wayland (`wayland-0`)
* **Audio Server**: PipeWire 1.6+ (PulseAudio compatibility interface)
* **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (6144 MiB VRAM)
* **Host RAM**: AMD Ryzen CPU, 24 GiB RAM
* **Python**: Python 3.12 managed via `uv`

---

## 3. Directory Layout

```
laya-desktop-agent/
├── app/                  # Daemon, CLI entrypoint, evdev Push-to-Talk listener
│   ├── cli.py            # Rich CLI interface (run, diagnose, exec, status, stop)
│   ├── daemon.py         # Main daemon with single-instance lock and signal handling
│   └── ptt_listener.py   # Evdev global key listener (F12 PTT, F11 Emergency Stop)
├── core/                 # Pipeline coordinator, state machine, event bus, config
│   ├── config.py         # Central Pydantic / YAML configuration
│   ├── coordinator.py    # Pipeline coordinator (Audio -> STT -> Laya -> Exec -> Verify -> TTS)
│   ├── events.py         # Internal pub/sub event bus
│   └── state.py          # Agent lifecycle state machine
├── models/               # Model wrappers and lifecycle management
│   ├── laya_engine.py    # Convai Laya decision router (ModernBERT 421M)
│   ├── manager.py        # Central model manager (LOAD -> WARM -> USE -> IDLE -> UNLOAD)
│   ├── piper_engine.py   # Piper TTS (CPU ONNX, 0 MiB VRAM)
│   └── whisper_engine.py # Faster-Whisper STT (small.en CUDA float16 / CPU int8)
├── tools/                # Controlled desktop tools registry (35+ tools)
│   ├── app_control.py    # open/close/switch applications & folders
│   ├── browser_control.py# browser search/navigate/download/click/fill (DOM)
│   ├── click_control.py  # click_element, select_option, submit, fill_form,
│   │                     # ui_action (any app verb), wait_action, verify_action
│   ├── file_control.py   # open/move/copy/rename/delete/compress/extract/find
│   ├── input_control.py  # scroll, type_text (AT-SPI/wtype/clipboard), key combos
│   ├── save_control.py   # save-as format conversion (Pillow native + GUI path)
│   ├── screen_control.py # read_screen (AT-SPI2), take_screenshot (Spectacle)
│   ├── system_control.py # set_volume, mute_volume, media_control (MPRIS)
│   └── window_control.py # minimize_window, maximize_window, switch_desktop
├── integrations/         # Native OS & Wayland interfaces
│   ├── atspi_client.py   # Accessibility tree reader + UI element finder
│   ├── dbus_client.py    # KDE Plasma notifications and MPRIS media control
│   ├── flatpak_helper.py # Desktop application indexer and launcher
│   ├── kdotool_client.py # Wayland window management wrapper
│   └── playwright_client.py # Browser automation (CDP attach → Chromium → xdg-open)
├── vision/               # Screen perception
│   └── screen_analyzer.py# AT-SPI element location + OCR fallback + activation
├── planning/             # NL understanding & autonomous execution
│   ├── semantic_parser.py# Generalized verb-frame semantic parser
│   ├── goal_executor.py  # Per-step observe→act→verify→recover loop
│   ├── goal_planner.py   # Decomposition, replanning, completeness checks
│   ├── llm_backend.py    # Rule (parser) / API / auto planner backends
│   └── execution_memory.py # Step history + referring-expression context
├── execution/            # Parameter binder and timeout-managed action executor
├── verification/         # Cached observer, honest verifier, GoalVerifier, recovery
├── safety/               # 3-tier safety policy and global emergency stop latch
├── monitoring/           # Real-time NVML VRAM guard, cache reaper, latency tracker
├── tests/                # Automated pytest test suite and VRAM stress test
├── scripts/              # Bash runners (start, stop, diagnose, test, stress_test)
├── configs/              # Profiles: default (balanced), low_resource, performance
├── cache/                # Isolated model, audio, screenshot, and runtime caches
├── logs/                 # Rolling log files and metrics.jsonl
└── docs/                 # Architectural deep dives, command reference, safety docs
```

---

## 4. Quickstart

### Automated Environment Setup
```bash
cd ~/Projects/laya-desktop-agent
bash scripts/setup_env.sh
```

### 1. Check System Status & Health
```bash
./scripts/start.sh status
# or
.venv/bin/python -m app.cli status
```
Outputs live VRAM, RAM, GPU temperature, and interface health checks (GPU, Audio, kdotool, DBus, AT-SPI2, Disk).

### 2. Test in Safe Diagnostic Mode (Zero OS Execution)
```bash
bash scripts/diagnose.sh
```
* Hold **F12**, speak: *"Open Google Chrome"* or *"Scroll down"*, release **F12**.
* The agent transcribes your speech, runs Laya inference, and prints the structured decision and planned action table **without** modifying your desktop.

### 3. Run a Single Command from Terminal
```bash
.venv/bin/python -m app.cli exec "Open Chrome"
.venv/bin/python -m app.cli exec "Take a screenshot"
.venv/bin/python -m app.cli exec "Set volume to 50 percent"
```

### 4. Start the Active Voice Agent Daemon
```bash
bash scripts/start.sh
```
* **Hold F12**: Speak any command. The agent transcribes, executes, verifies, and provides natural voice feedback.
* **Press F11**: Emergency Stop. Immediately halts all active tools, cancels queues, and mutes speech.

---

## 5. Systemd User Service

To run the agent as a persistent background daemon:

```bash
# Enable and start the user service
systemctl --user enable --now laya-voice-agent.service

# Check service status
systemctl --user status laya-voice-agent.service

# View live rolling logs
journalctl --user -u laya-voice-agent.service -f

# Stop service
systemctl --user stop laya-voice-agent.service
```

---

## 6. Natural Language Capabilities (generalized — NOT a command list)

The system understands goal-level natural language and derives the steps itself.
Every action is verified before success is reported; failures recover or fail honestly.

| Example utterance | Agent behavior | Verification |
|---|---|---|
| "Open Chrome." / "Can you open Chrome?" / "Launch Chrome for me" | Resolves app from live catalog → launches/switches, unminimizes, focuses | Active window matches target |
| "Open Chrome and Opera." / "Open Chrome, Steam and Dolphin." | Verb inheritance distributes over the coordinated list; each app opened & verified | All windows open |
| "Search for cats and dogs." | The conjunction stays INSIDE the query (never split) | Browser title/URL match |
| "Open Chrome, go to YouTube and search for XYZ." | Navigate → search-engine inherited from YouTube context → results opened | URL contains query on the right engine |
| "Open Chrome and search for Hello World." | Open + Google search | Verified in browser |
| "Open Opera and open WhatsApp." | Opera app + web.whatsapp.com | Both verified |
| "Open Upscayl, add this image, select 5x and proceed." | UI elements located by label via AT-SPI (OCR fallback) and activated | Activation logged + step verified |
| "Open Steam and open this game." | Referring expression resolved in Steam's UI | Element activation |
| "Open this image and save it as AVIF." | Native conversion (Pillow) with GUI save-dialog fallback | Output file exists |
| "Open Chrome, download this file, move it to Downloads, rename it and open it." | Full autonomous chain with referring-expression resolution across steps | Each step verified |
| "Move this file to Wallpapers." / "Create a folder called Projects." / "Find the largest files." / "Delete junk.txt." | Native filesystem ops (delete requires confirmation) | Filesystem assertions |
| "Type \"hello world\" and press enter." / "Press control w." | AT-SPI/wtype synthesis, combos, clipboard for long text | Input log + self-verification |
| "What is on my screen?" / "Take a screenshot" / "Set volume to 50%" / "Mute audio" / "Minimize this window" / "Scroll down twice" / "Switch to desktop two" | Direct capabilities | Verified per action type |

**Novel combinations work without new code** — the parser composes intents dynamically:
`'Launch Dolphin and open Steam then press escape.'` is a tested acceptance case.

---

## 7. Performance & Latency Benchmarks

Tested on Arch Linux, AMD Ryzen CPU, NVIDIA RTX 3050 Laptop GPU (6 GB VRAM):

* **Convai Laya Decision Hot-Path Latency**: **32.89 ms** on CUDA.
* **Faster-Whisper STT (small.en)**: ~120–160 ms on CUDA.
* **Piper TTS (en_US-lessac-medium)**: ~65–85 ms on CPU via ONNX Runtime (0 MiB VRAM).
* **End-to-End Voice Response Time**: **~280–400 ms** total latency.
* **30-Cycle Long-Run Stress Test**:
  - Process Host RSS Drift: **+0.07 MB** (70 KB across 30 full command cycles).
  - VRAM Steady State: Fully bounded (stable within PyTorch caching allocator pool).
  - Verdict: **PASSED — Zero progressive memory accumulation**.

---

## 8. Safety & Emergency Controls

1. **3-Tier Permission Gating**:
   - **SAFE**: Auto-executed (open apps, switch windows, scroll, read screen, volume).
   - **MODERATE**: Gated by configuration (moving files, modifying settings).
   - **HIGH-RISK**: Always blocked without explicit confirmation (file deletion, killing processes).
2. **Global Hardware Emergency Stop (`KEY_F11`)**:
   - Monitored globally at the kernel evdev level.
   - Instantly latches into `PAUSED` state.
   - Drops action queues, terminates active subprocesses, and immediately halts TTS audio.

---

## 9. Running Automated Tests

```bash
# Run full unit, integration, and acceptance E2E suite (122 passed)
bash scripts/test.sh
# or: .venv/bin/python -m pytest tests/ -q

# Acceptance E2E (real-user scenarios on a hermetic simulated desktop:
# fake window manager, fake apps, fake browser, fake AT-SPI screen state)
.venv/bin/python -m pytest tests/test_acceptance_e2e.py -v

# Run 30-cycle VRAM and RAM stress benchmark
bash scripts/stress_test.sh
```

The test suite covers:
- **Acceptance E2E (40 tests)**: every §22 scenario class — simple/multi-app
  conjunctions, browser flows, desktop UI automation (Upscayl/Steam workflows),
  file operations, recovery (injected transient failures, unresolvable apps,
  time budgets, watchdog cancellation, emergency stop), the voice pipeline
  (synthetic speech → VAD → STT → goal execution; silence rejection; Whisper
  hallucination filtering; STT error repair), and latency budgets
  (time-to-first-action, observation-cache subprocess-count assertion)
- WebRTC VAD silence filtering and tone detection
- End-to-end audio -> STT -> parser -> planner -> executor -> observer ->
  verifier -> TTS pipeline
- Kernel evdev Push-to-Talk and Emergency Stop lifecycle (skips cleanly
  without /dev/uinput; runs for real on the target machine)
- KDE Plasma 6 Wayland `kdotool` window search, activation, and desktop switching
- Semantic parser clause segmentation, verb inheritance, query protection
- 3-tier safety policy enforcement (file deletion gated)
- Single-instance `fcntl.flock` daemon locking
- Controlled tool registry execution
- Native AT-SPI2 keyboard/mouse input synthesis and wtype fallback
- NVML VRAM Guard 3-tier status transitions and emergency callbacks

All tests are **hermetic**: they run against the simulated desktop in
`tests/sim/` (fake window manager, applications, browser, and accessibility
screen state) — no dependency on the host's installed apps or hardware.

---

## 10. License

Apache 2.0. Built for personal, local-first, privacy-preserving desktop automation.
