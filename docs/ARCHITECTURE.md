# Architecture Documentation: Laya Desktop Agent

## System Architecture

The Laya Desktop Agent is an ultra-low-latency, privacy-preserving local voice control system for Arch Linux and KDE Plasma 6 (Wayland). It operates with strict resource bounds designed for systems with 6 GB VRAM.

```
+-----------------------------------------------------------------------------------+
|                                VOICE INPUT PIPELINE                               |
|                                                                                   |
|  [Microphone] -> [PipeWire 16kHz PCM] -> [WebRTC VAD] -> [Faster-Whisper STT]   |
+------------------------------------------+----------------------------------------+
                                           | Transcription
                                           v
+-----------------------------------------------------------------------------------+
|                        CONVAI LAYA DECISION & ROUTING ENGINE                      |
|                                                                                   |
|  [Desktop Context] + [Command Text] -> [ModernBERT-large 421M]                   |
|  Outputs: Intent (Choice) | Safety (Choice) | Confirm? (Noul) | Observe? (Noul)   |
+------------------------------------------+----------------------------------------+
                                           | Structured Routing Decision
                                           v
+-----------------------------------------------------------------------------------+
|                            SAFETY & EXECUTION SUBSYSTEM                           |
|                                                                                   |
|  [Emergency Stop Latch] -> [3-Tier Safety Policy] -> [Parameter Binder]           |
|                                         |                                         |
|                                         v                                         |
|  [Tool Registry]:                                                                 |
|   • kdotool (Window Switch, Close, Minimize, Maximize, Desktops)                  |
|   • AppLauncher (gtk-launch / gio for Native & Flatpak apps)                      |
|   • AT-SPI2 / Touchpoint (Accessibility UI Text Extraction)                       |
|   • BrowserClient (Playwright / Web Navigation & YouTube Search)                  |
|   • System Controls (PipeWire volume via pactl, MPRIS DBus Media)                 |
+------------------------------------------+----------------------------------------+
                                           | State Change
                                           v
+-----------------------------------------------------------------------------------+
|                           OBSERVATION & VERIFICATION LOOP                         |
|                                                                                   |
|  [State Observer] -> [Expected vs Observed Verification] -> [Bounded Recovery]    |
+------------------------------------------+----------------------------------------+
                                           | Result Text
                                           v
+-----------------------------------------------------------------------------------+
|                                 AUDIO FEEDBACK                                    |
|                                                                                   |
|  [Piper TTS (ONNX on CPU - 0 MiB VRAM)] -> [PipeWire Audio Sink]                  |
+-----------------------------------------------------------------------------------+
```

## Latency Profile

| Stage | Technology | Device | Average Latency |
|---|---|---|---|
| Audio Capture Buffer | sounddevice | CPU | ~5 ms |
| VAD Silence Trimming | WebRTC VAD | CPU | < 1 ms |
| Speech-to-Text | Faster-Whisper (small.en) | CUDA (float16) | ~120 - 180 ms |
| Intent & Routing | Convai Laya (ModernBERT) | CUDA (float16) | ~32 - 40 ms |
| Parameter Binding | Rule & Regex Entity Parser | CPU | < 1 ms |
| Desktop Execution | kdotool / gtk-launch / DBus | OS / Wayland | ~15 - 45 ms |
| State Observation | kdotool search / getactivewindow | Wayland | ~20 - 40 ms |
| Verification | Rule-based State Asserter | CPU | < 1 ms |
| Text-to-Speech | Piper TTS (lessac-medium) | CPU (ONNX) | ~60 - 90 ms |
| **Total End-to-End** | **Full Local Pipeline** | **Hybrid** | **~260 - 400 ms** |

## GPU Memory Allocation

Total GPU VRAM: 6144 MiB (NVIDIA GeForce RTX 3050 Laptop).

* Baseline Desktop (KWin + Plasma + Xwayland + Ghostty): ~630 MiB
* Laya Decision Engine (ModernBERT-large, fp16): ~840 MiB
* Faster-Whisper STT (small.en, float16): ~460 MiB
* Piper TTS: 0 MiB (Strictly pinned to CPU)
* **Agent Active Allocation**: ~1.30 GiB
* **Total System VRAM**: ~1.93 GiB (leaving > 4.2 GiB free for desktop apps, browsers, and user tasks).
