# Laya Voice Desktop Agent

> **Production-quality, fast, privacy-preserving local voice agent for Arch Linux + KDE Plasma 6 (Wayland) powered by Convai Laya (ModernBERT-large 421M non-autoregressive decision engine) with strict VRAM protection.**

---

## 1. Overview

The **Laya Desktop Agent** allows natural voice control of your Arch Linux desktop with zero cloud dependency. Every component—speech recognition, decision routing, safety gating, OS execution, state verification, and text-to-speech—runs 100% locally on your machine.

### Key Pipeline
$$\text{Microphone} \longrightarrow \text{WebRTC VAD} \longrightarrow \text{Faster-Whisper STT} \longrightarrow \text{Convai Laya Router} \longrightarrow \text{Tool Executor} \longrightarrow \text{State Observer} \longrightarrow \text{Verifier} \longrightarrow \text{Piper TTS}$$

* **Sub-35ms Decision Routing**: Uses Convai Laya (`laya.Router`), a non-autoregressive "System 1" model that evaluates intent, safety tiers, and observation requirements in a single forward pass (~32.89 ms hot-path inference on CUDA).
* **Strict VRAM Guardrails**: Built for a 6 GB VRAM budget (RTX 3050). The agent never exceeds ~1.3–1.9 GB of VRAM, preserving >4.2 GB of free headroom for your desktop, browser, and games.
* **Wayland & KDE Plasma 6 Native**: Uses `/usr/local/bin/kdotool` for window manipulation, AT-SPI2 (`PyGObject`) for UI accessibility inspection, FreeDesktop/KDE DBus for notifications and MPRIS media playback, and `gtk-launch` for native/flatpak applications.
* **Push-to-Talk First**: Global hardware key capture via Linux `evdev` (`KEY_F12`), completely bypassing Wayland key-grab restrictions with zero input latency.
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
├── tools/                # Controlled desktop tools registry
│   ├── app_control.py    # open_application, close_application, switch_window
│   ├── browser_control.py# browser_search (YouTube/Google), browser_navigate
│   ├── input_control.py  # scroll (up/down), type_text, press_key
│   ├── screen_control.py # read_screen (AT-SPI2), take_screenshot (Spectacle)
│   ├── system_control.py # set_volume, mute_volume, media_control (MPRIS)
│   └── window_control.py # minimize_window, maximize_window, switch_desktop
├── integrations/         # Native OS & Wayland interfaces
│   ├── atspi_client.py   # Accessibility tree reader via PyGObject
│   ├── dbus_client.py    # KDE Plasma notifications and MPRIS media control
│   ├── flatpak_helper.py # Desktop application indexer and launcher
│   ├── kdotool_client.py # Wayland window management wrapper
│   └── playwright_client.py # Browser DOM interaction client
├── execution/            # Parameter binder and timeout-managed action executor
├── verification/         # Desktop state observer, verifier, and recovery manager
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

## 6. Supported Voice Commands

| Voice Command | Action Executed | Verification Method |
|---|---|---|
| **"Open Chrome"** | Launch or switch to Google Chrome | Asserts Chrome window is active in window stack |
| **"Open Telegram"** | Launch or switch to Telegram Desktop | Asserts Telegram window is active |
| **"Close this window"** | Closes currently focused window | Asserts window ID is removed from KWin |
| **"Switch to Opera"** | Focuses and brings Opera to foreground | Asserts active window matches Opera |
| **"Scroll down"** | Sends Page_Down scroll event | Tool execution acknowledged |
| **"Scroll up twice"** | Sends Page_Up scroll event twice | Tool execution acknowledged |
| **"Search YouTube for Linux tutorials"** | Opens YouTube search in browser | Browser URL navigation |
| **"Take a screenshot"** | Captures screen via Spectacle | Asserts PNG created in cache, auto-cleaned |
| **"What is on my screen?"** | Reads active UI text via AT-SPI2 | Extracted structured text spoken via TTS |
| **"Set volume to 50 percent"** | Sets PipeWire master sink to 50% | PipeWire volume adjusted |
| **"Mute audio"** | Toggles audio mute | PipeWire mute toggled |
| **"Minimize this window"** | Minimizes active window | Asserts active window minimized |
| **"Maximize this window"** | Maximizes active window | Asserts active window maximized |
| **"Switch to desktop two"** | Switches KDE virtual desktop | Asserts desktop number == 2 |

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
# Run full unit and integration test suite (10/10 passed)
bash scripts/test.sh

# Run 30-cycle VRAM and RAM stress benchmark
bash scripts/stress_test.sh
```

---

## 10. License

Apache 2.0. Built for personal, local-first, privacy-preserving desktop automation.
