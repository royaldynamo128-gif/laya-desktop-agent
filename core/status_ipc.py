"""
IPC Status Broadcaster and Client for Laya Desktop Agent.
Uses Unix Domain Sockets (/run/user/1000/laya_status.sock) to stream
live state transitions and 'Next Action' instructions to the Wayland status overlay.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
import json
import logging
import os
import socket
import threading
import time

logger = logging.getLogger(__name__)


def get_default_socket_path() -> Path:
    runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    return runtime_dir / "laya_status.sock"


@dataclass
class StatusPayload:
    state: str            # e.g. READY, LISTENING, TRANSCRIBING, UNDERSTANDING, EXECUTING, VERIFYING, SUCCESS, FAILURE, ERROR
    title: str            # e.g. "🎤 Voice Agent Ready", "● LISTENING"
    detail: str           # e.g. "Speak now", "Open Chrome", "Transcription..."
    next_action: str      # e.g. "Hold Meta + Alt to speak", "Release Meta + Alt when finished"
    level: str = "info"   # info, warning, error, success
    command_text: str = ""
    timestamp: float = 0.0

    def to_json(self) -> str:
        d = asdict(self)
        if not self.timestamp:
            d["timestamp"] = time.time()
        return json.dumps(d)

    @classmethod
    def from_json(cls, data: str) -> "StatusPayload":
        d = json.loads(data)
        return cls(**d)


class StatusBroadcaster:
    """Unix Domain Socket server hosting live status stream for the desktop overlay."""

    def __init__(self, socket_path: Optional[Path] = None):
        self.socket_path = socket_path or get_default_socket_path()
        self._server_sock: Optional[socket.socket] = None
        self._clients: List[socket.socket] = []
        self._lock = threading.Lock()
        self._running = False
        self._server_thread: Optional[threading.Thread] = None

        # Retain last status to hydrate newly connected clients immediately
        self._current_status = StatusPayload(
            state="READY",
            title="🎤 Voice Agent Ready",
            detail="Local agent warmed on CUDA",
            next_action="Hold Meta + Alt to speak",
            level="info",
            timestamp=time.time(),
        )

    @property
    def current_status(self) -> StatusPayload:
        with self._lock:
            return self._current_status

    def start(self) -> bool:
        """Start socket server thread."""
        try:
            if self.socket_path.exists():
                self.socket_path.unlink()

            self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._server_sock.bind(str(self.socket_path))
            self._server_sock.listen(10)
            self._running = True

            self._server_thread = threading.Thread(
                target=self._accept_loop,
                daemon=True,
                name="StatusSocketServer",
            )
            self._server_thread.start()
            logger.info("Status Broadcaster listening at: %s", self.socket_path)
            return True
        except Exception as e:
            logger.error("Failed to start Status Broadcaster on %s: %s", self.socket_path, e)
            return False

    def _accept_loop(self) -> None:
        while self._running:
            try:
                client_sock, _ = self._server_sock.accept()
                with self._lock:
                    self._clients.append(client_sock)
                    # Send initial snapshot immediately
                    try:
                        msg = self._current_status.to_json() + "\n"
                        client_sock.sendall(msg.encode("utf-8"))
                    except Exception as e:
                        logger.debug("Failed to send initial status snapshot: %s", e)
            except Exception:
                if not self._running:
                    break

    def broadcast(
        self,
        state: str,
        title: str,
        detail: str,
        next_action: str,
        level: str = "info",
        command_text: str = "",
    ) -> None:
        """Broadcast state update to all connected overlay clients."""
        payload = StatusPayload(
            state=state.upper(),
            title=title,
            detail=detail,
            next_action=next_action,
            level=level,
            command_text=command_text,
            timestamp=time.time(),
        )

        with self._lock:
            self._current_status = payload
            encoded = (payload.to_json() + "\n").encode("utf-8")
            stale_clients = []

            for client in self._clients:
                try:
                    client.sendall(encoded)
                except Exception:
                    stale_clients.append(client)

            for stale in stale_clients:
                try:
                    stale.close()
                except Exception:
                    pass
                if stale in self._clients:
                    self._clients.remove(stale)

    def stop(self) -> None:
        """Stop server and clean up socket."""
        self._running = False
        with self._lock:
            for client in self._clients:
                try:
                    client.close()
                except Exception:
                    pass
            self._clients.clear()

        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
            self._server_sock = None

        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except Exception:
                pass


class StatusClient:
    """Client for listening to daemon status events."""

    def __init__(
        self,
        on_status: Callable[[StatusPayload], None],
        socket_path: Optional[Path] = None,
    ):
        self.socket_path = socket_path or get_default_socket_path()
        self.on_status = on_status
        self._sock: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._connection_loop,
            daemon=True,
            name="StatusClientThread",
        )
        self._thread.start()

    def _connection_loop(self) -> None:
        while self._running:
            try:
                if not self.socket_path.exists():
                    time.sleep(0.5)
                    continue

                self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self._sock.connect(str(self.socket_path))
                logger.debug("Connected to daemon status socket at: %s", self.socket_path)

                buffer = ""
                while self._running:
                    chunk = self._sock.recv(4096)
                    if not chunk:
                        break
                    buffer += chunk.decode("utf-8", errors="replace")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if line:
                            try:
                                payload = StatusPayload.from_json(line)
                                self.on_status(payload)
                            except Exception as e:
                                logger.debug("Error parsing status line: %s", e)

            except Exception as e:
                logger.debug("StatusClient connection error (retrying): %s", e)
            finally:
                if self._sock:
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                    self._sock = None

            if self._running:
                time.sleep(1.0)

    def stop(self) -> None:
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
