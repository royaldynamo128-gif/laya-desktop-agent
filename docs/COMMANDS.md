# Voice Command Catalog: Laya Desktop Agent

The agent supports natural phrasing for a wide variety of desktop actions on Arch Linux + KDE Plasma 6.

## 1. Application Control

| Voice Command | Action | Implementation |
|---|---|---|
| "Open Chrome" | Launch or switch to Google Chrome | `gtk-launch` / `kdotool search` |
| "Open Telegram" | Launch or switch to Telegram (Flatpak) | `gtk-launch org.telegram.desktop.desktop` |
| "Open Terminal" | Launch or switch to Ghostty / Konsole | `gtk-launch com.mitchellh.ghostty.desktop` |
| "Open my Projects folder" | Open Dolphin file manager in `~/Projects` | `xdg-open /home/rai/Projects` |
| "Close this window" | Gracefully close focused window | `kdotool windowclose` |
| "Switch to Opera" | Bring Opera window to foreground | `kdotool windowactivate` |
| "Switch to previous window"| Focus previous window in stack | `kdotool search` + index 1 |

## 2. Window Manipulation

| Voice Command | Action | Implementation |
|---|---|---|
| "Minimize this window" | Minimize currently active window | `kdotool windowminimize` |
| "Maximize this window" | Maximize currently active window | `kdotool windowstate --add MAXIMIZED` |
| "Switch to desktop two" | Change KDE virtual desktop to desktop 2 | `kdotool set_desktop 2` |

## 3. Web & Browser

| Voice Command | Action | Implementation |
|---|---|---|
| "Search YouTube for Linux tutorials" | Open YouTube search results | `xdg-open https://www.youtube.com/...` |
| "Search Google for Arch Linux" | Open Google search results | `xdg-open https://www.google.com/...` |
| "Navigate to github.com" | Open GitHub in default browser | `xdg-open https://github.com` |

## 4. Input & Scrolling

| Voice Command | Action | Implementation |
|---|---|---|
| "Scroll down" | Scroll page down one step | `wtype -k Page_Down` |
| "Scroll up twice" | Scroll page up two steps | `wtype -k Page_Up` x2 |
| "Press Enter" | Press Return key in focused window | `wtype -k Return` |
| "Press Escape" | Press Escape key | `wtype -k Escape` |
| "Type this message: Hello world" | Type string into focused input field | `wtype "Hello world"` |

## 5. Screen & Inspection

| Voice Command | Action | Implementation |
|---|---|---|
| "Take a screenshot" | Capture full desktop screenshot | `spectacle -b -n -o <path>` |
| "What is on my screen?" | Read active UI text elements | AT-SPI2 accessibility tree |
| "Read this page" | Read focused window content | AT-SPI2 text interface |

## 6. System & Audio

| Voice Command | Action | Implementation |
|---|---|---|
| "Set volume to 50 percent" | Set PipeWire master volume to 50% | `pactl set-sink-volume @DEFAULT_SINK@ 50%` |
| "Volume up" | Increase volume by 10% | `pactl set-sink-volume @DEFAULT_SINK@ +10%` |
| "Volume down" | Decrease volume by 10% | `pactl set-sink-volume @DEFAULT_SINK@ -10%` |
| "Mute audio" | Toggle mute state | `pactl set-sink-mute @DEFAULT_SINK@ toggle` |
| "Play music" / "Pause music"| Toggle MPRIS media player playback | DBus `org.mpris.MediaPlayer2` |
