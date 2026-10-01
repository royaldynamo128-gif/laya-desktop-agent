"""
Fake AT-SPI accessibility provider for hermetic UI-automation tests.

Exposes a scripted screen state (per-application element lists) that the
ScreenAnalyzer consumes exactly like the real AT-SPI2 tree. Element
activations are logged for assertions.

Screen state file ($ATSPI_SCREEN_STATE, JSON):
{
  "apps": {
    "<wm_class>": {
      "elements": [
        {"name": "Add Image", "role": "push button", "x": 10, "y": 20, "width": 90, "height": 24, "actions": ["click"]},
        ...
      ]
    }
  }
}

The active application is derived from the simulated window manager state
($KDOTOOL_STATE), so UI scripts follow real window focus.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import difflib


def _screen_state_path() -> str:
    return os.environ.get("ATSPI_SCREEN_STATE", "/tmp/fake_atspi_state.json")


def _action_log_path() -> str:
    return os.environ.get("ATSPI_ACTION_LOG", "/tmp/fake_atspi_actions.jsonl")


class FakeExtents:
    def __init__(self, x, y, width, height):
        self.x, self.y, self.width, self.height = x, y, width, height


class FakeComponent:
    def __init__(self, accessible):
        self._acc = accessible

    def grab_focus(self) -> bool:
        return True


class FakeAccessible:
    """Mimics the AT-SPI accessible object interface used by ScreenAnalyzer."""

    def __init__(self, element: Dict[str, Any], log_fn):
        self._element = element
        self._log = log_fn

    def get_name(self) -> str:
        return self._element.get("name", "")

    def get_role_name(self) -> str:
        return self._element.get("role", "")

    def get_n_actions(self) -> int:
        return len(self._element.get("actions", ["click"]))

    def get_action_name(self, i: int) -> str:
        actions = self._element.get("actions", ["click"])
        return actions[i] if i < len(actions) else "click"

    def do_action(self, i: int) -> bool:
        action = self.get_action_name(i)
        self._log("activate", self._element.get("name", ""), action, self._element.get("role", ""))
        return True

    def get_extents(self) -> FakeExtents:
        return FakeExtents(
            self._element.get("x", 0),
            self._element.get("y", 0),
            self._element.get("width", 80),
            self._element.get("height", 24),
        )

    def get_component_iface(self) -> FakeComponent:
        return FakeComponent(self)

    def get_child_count(self) -> int:
        return 0

    def get_state_set(self):
        class _S:
            def contains(self, _t) -> bool:
                return False

        return _S()

    def get_text_iface(self):
        return None


class FakeAtspi:
    """Drop-in AtspiClient replacement for the simulated desktop."""

    def __init__(self, screen_state_path: Optional[str] = None, action_log_path: Optional[str] = None):
        self.enabled = True
        self._screen_path = screen_state_path or _screen_state_path()
        self._action_log = action_log_path or _action_log_path()

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def _load_screen(self) -> Dict[str, Any]:
        try:
            with open(self._screen_path) as f:
                return json.load(f)
        except Exception:
            return {"apps": {}}

    def _active_app_class(self) -> str:
        try:
            with open(os.environ.get("KDOTOOL_STATE", "")) as f:
                state = json.load(f)
            active = state.get("active")
            if active and active in state.get("windows", {}):
                return state["windows"][active].get("class", "").lower()
        except Exception:
            pass
        return ""

    def _log_action(self, kind: str, name: str, action: str, role: str) -> None:
        entry = {"kind": kind, "element": name, "action": action, "role": role}
        try:
            with open(self._action_log, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # AtspiClient interface
    # ------------------------------------------------------------------

    def get_desktop(self):
        return None

    def get_active_app_text(self, max_elements: int = 50) -> List[str]:
        app_class = self._active_app_class()
        screen = self._load_screen()
        app = screen.get("apps", {}).get(app_class, {})
        texts = [e.get("name", "") for e in app.get("elements", []) if e.get("name")]
        return texts[:max_elements]

    def find_elements_by_label(self, label: str, max_results: int = 8) -> List[Dict[str, Any]]:
        app_class = self._active_app_class()
        screen = self._load_screen()
        # Search the active app first, then all apps (fallback like real AT-SPI)
        app_list = []
        apps = screen.get("apps", {})
        if app_class in apps:
            app_list.append(apps[app_class])
        for k, v in apps.items():
            if k != app_class:
                app_list.append(v)

        target = label.strip().lower()
        results: List[Dict[str, Any]] = []
        for app in app_list:
            for elem in app.get("elements", []):
                name = elem.get("name", "")
                if not name:
                    continue
                v = name.lower()
                exact = v == target
                substr = target in v or v in target
                ratio = difflib.SequenceMatcher(None, target, v).ratio()
                t_tokens = set(re.findall(r"\w+", target))
                v_tokens = set(re.findall(r"\w+", v))
                subset = bool(t_tokens) and t_tokens.issubset(v_tokens)
                if exact or substr or ratio >= 0.82 or subset:
                    conf = 1.0 if exact else (0.9 if subset or substr else ratio)
                    results.append({
                        "name": name,
                        "role": elem.get("role", ""),
                        "x": elem.get("x", 0),
                        "y": elem.get("y", 0),
                        "width": elem.get("width", 80),
                        "height": elem.get("height", 24),
                        "accessible": FakeAccessible(elem, self._log_action),
                        "confidence": round(conf, 3),
                    })
        results.sort(key=lambda r: r["confidence"], reverse=True)
        return results[:max_results]

    def get_editable_fields(self, max_fields: int = 10) -> List[Dict[str, Any]]:
        app_class = self._active_app_class()
        screen = self._load_screen()
        app = screen.get("apps", {}).get(app_class, {})
        fields = []
        for elem in app.get("elements", []):
            if elem.get("role") in ("text", "entry", "search", "text box"):
                fields.append({
                    "name": elem.get("name", ""),
                    "role": elem.get("role"),
                    "x": elem.get("x", 0), "y": elem.get("y", 0),
                    "width": elem.get("width", 80), "height": elem.get("height", 24),
                    "accessible": FakeAccessible(elem, self._log_action),
                })
        return fields[:max_fields]

    def find_elements_by_role(self, role: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Role-based element search in the active application (like real AT-SPI)."""
        app_class = self._active_app_class()
        screen = self._load_screen()
        apps = screen.get("apps", {})
        app_list = []
        if app_class in apps:
            app_list.append(apps[app_class])
        results: List[Dict[str, Any]] = []
        target_role = (role or "").strip().lower()
        for app in app_list:
            for elem in app.get("elements", []):
                if (elem.get("role", "") or "").lower() == target_role:
                    results.append({
                        "name": elem.get("name", ""),
                        "role": elem.get("role", ""),
                        "x": elem.get("x", 0),
                        "y": elem.get("y", 0),
                        "width": elem.get("width", 80),
                        "height": elem.get("height", 24),
                        "accessible": FakeAccessible(elem, self._log_action),
                        "confidence": 1.0,
                    })
        return results[:max_results]


def write_screen_state(apps: Dict[str, Any], path: Optional[str] = None) -> None:
    p = path or _screen_state_path()
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"apps": apps}, f)
    os.replace(tmp, p)


def read_action_log(path: Optional[str] = None) -> List[Dict[str, Any]]:
    p = path or _action_log_path()
    try:
        with open(p) as f:
            return [json.loads(line) for line in f if line.strip()]
    except Exception:
        return []
