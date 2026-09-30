# VRAM Management & Memory Safety Policy

## 1. Hardware Budget & Guardrails

The Laya Desktop Agent is purpose-built for constrained GPU environments, specifically tuned for an **NVIDIA GeForce RTX 3050 Laptop GPU (6144 MiB VRAM)**.

### Memory Ceiling
* **Maximum Agent VRAM Allocation**: 2048 MiB (in `BALANCED` profile).
* **Guaranteed System Headroom**: > 4000 MiB reserved for KDE Plasma, Xwayland, browsers, and user applications.
* **Hard Upper Ceiling**: The agent will never intentionally consume more than 2.0 GiB of VRAM under normal operation.

---

## 2. Dynamic Memory Tiers

| Tier | Utilization Trigger | System Behavior |
|---|---|---|
| **NORMAL** | `< 75.0%` GPU VRAM | Full GPU acceleration active for Faster-Whisper (float16) and Convai Laya (fp16). Piper TTS pinned to CPU. |
| **WARNING** | `75.0% - 85.0%` GPU VRAM | Rejects any new model allocations on GPU. Flags warning logs. Decreases idle unload timeout to 120s. |
| **CRITICAL** | `> 85.0%` GPU VRAM | Executes emergency cleanup: unloads Faster-Whisper and Laya from GPU, switches inference to CPU (`int8`), flushes PyTorch CUDA cache (`torch.cuda.empty_cache()`), runs Python `gc.collect()`. |

---

## 3. Component Memory Footprint

| Component | Model / Engine | Device | VRAM Footprint | Host RAM (RSS) |
|---|---|---|---|---|
| Decision Engine | Convai Laya (ModernBERT-large 421M) | CUDA (float16) | ~840 MiB | ~450 MiB |
| Speech-to-Text | Faster-Whisper (small.en) | CUDA (float16) | ~460 MiB | ~320 MiB |
| Text-to-Speech | Piper TTS (en_US-lessac-medium) | **CPU strictly (ONNX)** | **0 MiB** | ~75 MiB |
| Audio Capture | SoundDevice / PipeWire Ring Buffer | CPU | **0 MiB** | ~10 MiB |
| Window Manager | kdotool / DBus / AT-SPI2 | OS Process | **0 MiB** | ~25 MiB |
| **Total Agent** | **Active Pipeline** | **Hybrid** | **~1300 MiB** | **~880 MiB** |

---

## 4. Leak Prevention & Steady-State Verification

### Automated Lifecycle
1. **Model Idling**: Models are automatically evicted from GPU VRAM after `idle_timeout_sec` (default: 300s of inactivity).
2. **Audio Buffer Recycling**: Microsecond cleanup of raw PCM memory; audio streams are terminated and reopened per PTT session.
3. **Temporary File Reaping**: Screenshots and temporary audio WAV files are evicted immediately post-verification or after 30–60s TTL by the background `CacheReaper`.

### Stress Test Results (30 Continuous Cycles)
* **Initial Post-Warmup Baseline**: 4126.1 MB VRAM (including system background processes).
* **Final Post-Test VRAM**: 4215.1 MB VRAM (steady-state allocator pool delta: +88 MB).
* **Net Host RSS Delta**: **+0.07 MB** (70 KB across 30 full command cycles).
* **Verdict**: **PASSED — Zero progressive memory accumulation**.
