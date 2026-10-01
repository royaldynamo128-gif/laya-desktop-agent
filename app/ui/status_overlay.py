"""
Wayland Native Status Overlay for Laya Desktop Agent.
Uses GTK 4 Layer Shell to render a non-intrusive, zero-focus-stealing
live status pill in the bottom-right corner of the screen.
"""

import ctypes.util
import json
import logging
import os
import signal
import sys
import threading
import time

# Ensure gtk4-layer-shell is linked before libwayland-client via LD_PRELOAD
_LS = ctypes.util.find_library("gtk4-layer-shell") or "/usr/lib/libgtk4-layer-shell.so"
if _LS and os.path.exists(_LS) and _LS not in os.environ.get("LD_PRELOAD", ""):
    os.environ["LD_PRELOAD"] = (_LS + " " + os.environ.get("LD_PRELOAD", "")).strip()
    os.execv(sys.executable, [sys.executable] + sys.argv)

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Gtk4LayerShell", "1.0")
from gi.repository import Gtk, Gdk, GLib, Gtk4LayerShell

from core.status_ipc import StatusClient, StatusPayload, get_default_socket_path

logger = logging.getLogger("status_overlay")

# CSS Styling for Sleek Modern Overlay
CSS = b"""
window.background {
    background-color: transparent;
}

.status-pill {
    background-color: rgba(20, 20, 30, 0.92);
    border: 1px solid rgba(137, 180, 250, 0.3);
    border-radius: 12px;
    padding: 6px 12px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    min-width: 180px;
    max-width: 320px;
}

.status-pill.listening {
    border-color: rgba(243, 139, 168, 0.8);
    background-color: rgba(35, 18, 28, 0.95);
}

.status-pill.success {
    border-color: rgba(166, 227, 161, 0.8);
    background-color: rgba(18, 32, 24, 0.95);
}

.status-pill.failure, .status-pill.error {
    border-color: rgba(243, 139, 168, 0.85);
    background-color: rgba(35, 18, 22, 0.95);
}

.title-label {
    font-size: 11.5px;
    font-weight: 700;
    color: #cdd6f4;
}

.title-label.ready {
    color: #89b4fa;
}

.title-label.listening {
    color: #f38ba8;
}

.title-label.transcribing, .title-label.understanding, .title-label.processing {
    color: #cba6f7;
}

.title-label.executing, .title-label.verifying {
    color: #f9e2af;
}

.title-label.success {
    color: #a6e3a1;
}

.title-label.failure, .title-label.error {
    color: #f38ba8;
}

.detail-label {
    font-size: 10px;
    color: #a6adc8;
    margin-top: 1px;
}
"""


class StatusOverlayApp(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="org.laya.voice.status_overlay")
        self.window: Optional[Gtk.ApplicationWindow] = None
        self.pill_box: Optional[Gtk.Box] = None
        self.title_label: Optional[Gtk.Label] = None
        self.detail_label: Optional[Gtk.Label] = None
        self.next_action_label: Optional[Gtk.Label] = None
        self.status_client: Optional[StatusClient] = None

        self._watchdog_timer_id: Optional[int] = None
        self._current_state = "READY"

    def do_activate(self):
        if not self.window:
            self._setup_window()
            self._setup_css()
            self._start_ipc_client()

        # Unobtrusive idle: NEVER show window on startup or when idle
        self.window.set_visible(False)

    def _setup_window(self):
        self.window = Gtk.ApplicationWindow(application=self)
        self.window.set_title("Laya Status")

        # Initialize Wayland Layer Shell
        Gtk4LayerShell.init_for_window(self.window)
        Gtk4LayerShell.set_layer(self.window, Gtk4LayerShell.Layer.TOP)
        Gtk4LayerShell.set_anchor(self.window, Gtk4LayerShell.Edge.BOTTOM, True)
        Gtk4LayerShell.set_anchor(self.window, Gtk4LayerShell.Edge.RIGHT, True)
        Gtk4LayerShell.set_margin(self.window, Gtk4LayerShell.Edge.BOTTOM, 20)
        Gtk4LayerShell.set_margin(self.window, Gtk4LayerShell.Edge.RIGHT, 20)
        
        # NEVER steal keyboard focus
        Gtk4LayerShell.set_keyboard_mode(self.window, Gtk4LayerShell.KeyboardMode.NONE)

        # Main Pill Box
        self.pill_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self.pill_box.add_css_class("status-pill")

        # Row 1: Title (e.g. 🎙 Listening, ◌ Processing, ✓ Done)
        self.title_label = Gtk.Label(label="", xalign=0.0)
        self.title_label.add_css_class("title-label")
        self.pill_box.append(self.title_label)

        # Row 2: Detail (e.g. Speak now, Opening Chrome, error msg)
        self.detail_label = Gtk.Label(label="", xalign=0.0)
        self.detail_label.add_css_class("detail-label")
        self.pill_box.append(self.detail_label)

        self.window.set_child(self.pill_box)
        self.window.set_visible(False)

    def _setup_css(self):
        provider = Gtk.CssProvider()
        provider.load_from_data(CSS)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

    def _start_ipc_client(self):
        self.status_client = StatusClient(on_status=self._on_status_received)
        self.status_client.start()

    def _on_status_received(self, payload: StatusPayload):
        GLib.idle_add(self._update_ui_state, payload)

    def _update_ui_state(self, payload: StatusPayload):
        state = payload.state.upper()
        self._current_state = state

        # Cancel any previous auto-reset timer
        if self._watchdog_timer_id is not None:
            GLib.source_remove(self._watchdog_timer_id)
            self._watchdog_timer_id = None

        # When IDLE or READY: hide overlay completely
        if state in ("IDLE", "READY"):
            self._hide_overlay()
            return GLib.SOURCE_REMOVE

        # Reset css classes
        for cls in ["listening", "success", "failure", "error"]:
            self.pill_box.remove_css_class(cls)
        for cls in ["ready", "listening", "transcribing", "understanding", "processing", "executing", "verifying", "success", "failure", "error"]:
            self.title_label.remove_css_class(cls)

        # Apply state styling
        state_cls = state.lower()
        self.title_label.add_css_class(state_cls)
        if state in ("LISTENING", "SUCCESS", "FAILURE", "ERROR"):
            self.pill_box.add_css_class(state_cls)

        # Update text
        self.title_label.set_text(payload.title)
        if payload.detail:
            self.detail_label.set_text(payload.detail)
            self.detail_label.set_visible(True)
        else:
            self.detail_label.set_visible(False)

        # Make visible only during active operation
        if self.window:
            self.window.set_visible(True)
            self.window.present()

        # Auto-dismiss logic: return to unobtrusive hidden state
        if state == "SUCCESS":
            self._watchdog_timer_id = GLib.timeout_add(2500, self._hide_overlay)
        elif state in ("FAILURE", "ERROR"):
            self._watchdog_timer_id = GLib.timeout_add(3500, self._hide_overlay)
        elif state in ("LISTENING", "TRANSCRIBING", "UNDERSTANDING", "PROCESSING", "EXECUTING", "VERIFYING"):
            # 8-second safety watchdog in case pipeline terminates unexpectedly
            self._watchdog_timer_id = GLib.timeout_add(8000, self._hide_overlay)

        return GLib.SOURCE_REMOVE

    def _hide_overlay(self):
        if self._watchdog_timer_id is not None:
            GLib.source_remove(self._watchdog_timer_id)
            self._watchdog_timer_id = None
        self._current_state = "IDLE"
        if self.window:
            self.window.set_visible(False)
        return GLib.SOURCE_REMOVE


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    app = StatusOverlayApp()

    # Graceful exit on SIGINT / SIGTERM
    def signal_handler(sig, frame):
        if app.status_client:
            app.status_client.stop()
        app.quit()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    return app.run(None)


if __name__ == "__main__":
    sys.exit(main())
