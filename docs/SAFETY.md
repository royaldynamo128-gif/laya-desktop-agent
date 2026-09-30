# Safety & Emergency Abort Architecture

## 1. Three-Tier Safety Hierarchy

The agent enforces a strict 3-tier permission policy before any action reaches the execution layer. Natural language instructions are never passed directly to a raw shell.

```
       [Voice Command]
              │
              ▼
   [Laya Decision Engine]
              │
              ▼
  ┌───────────────────────┐
  │ Safety Level Assessed │
  └───────────┬───────────┘
              │
      ┌───────┴───────┬──────────────┐
      ▼               ▼              ▼
   [SAFE]        [MODERATE]     [HIGH-RISK]
      │               │              │
      │               │              ▼
      │               │     [User Confirmation]
      │               │     (Voice or Prompt)
      │               │              │
      ▼               ▼              ▼
  [Execute]   [Check Policy]    [Pass/Reject]
```

### SAFE Tier (Auto-Executed)
* Open application (Chrome, Telegram, Ghostty)
* Switch window
* Minimize / Maximize window
* Scroll up / down
* View screen / Read screen
* Take screenshot
* Adjust volume / media playback

### MODERATE Tier (Gated by Policy)
* Moving or renaming files
* Sending messages
* Modifying non-critical settings
* Gated by `SafetyConfig.require_confirmation_moderate` (default: false for trusted user).

### HIGH-RISK Tier (Always Requires Explicit Confirmation)
* Deleting files or directories
* Terminating or killing processes
* Destructive shell commands
* Modifying system security or sudo configurations
* Always blocked unless explicitly confirmed by the user.

---

## 2. Global Emergency Stop (E-Stop)

### Hardware Key Trigger
The agent listens directly to Linux input subsystem events via `evdev` (`/dev/input/event*`).
* **Emergency Stop Key**: `KEY_F11` (or `Ctrl+Alt+Escape`)
* Works globally across all Wayland applications.

### Immediate Abort Sequence
When `KEY_F11` is pressed:
1. **Hardware Latched**: `EmergencyStop.trigger()` immediately locks the system into `PAUSED` state.
2. **Audio Halted**: Any active Text-to-Speech playback via SoundDevice is immediately stopped.
3. **Recording Aborted**: In-progress microphone buffer is cleared.
4. **Queue Discarded**: All queued in-flight actions are cancelled and dropped.
5. **Desktop Notification**: Sends high-priority KDE notification: *"EMERGENCY STOP ACTIVATED"*.
6. **Latching**: The agent rejects any subsequent voice commands until explicitly cleared or restarted.
