"""
Parameter Binder for Laya Desktop Agent.
Extracts target entities and constructs validated tool arguments from transcription and Laya decisions.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import logging
import re
from models.laya_engine import LayaDecision
from integrations.app_catalog import AppCatalog

logger = logging.getLogger(__name__)


@dataclass
class BoundAction:
    tool_name: str
    arguments: Dict[str, Any]
    target: str = ""


class ParameterBinder:
    def __init__(self, catalog: Optional[AppCatalog] = None):
        self.catalog = catalog or AppCatalog()

    def _is_folder_request(self, target: str) -> bool:
        clean = re.sub(r"^(my|the)\s+", "", target.lower().strip())
        clean = re.sub(r"\s+(folder|directory)$", "", clean).strip()
        known_folders = {
            "projects", "project", "downloads", "download", "documents",
            "document", "desktop", "home", "pictures", "picture", "music",
            "videos", "video"
        }
        return clean in known_folders or target.lower().endswith("folder") or target.lower().endswith("directory")

    def bind(self, transcription: str, decision: Optional[LayaDecision] = None) -> Optional[BoundAction]:
        """Map speech transcription and Laya decision to a specific tool and arguments."""
        intent = decision.intent if decision else "other"
        text = transcription.strip()
        lower = text.lower()
        # Direct explicit keyword overrides
        # Normalize concatenated command verbs (e.g. 'openupskill' -> 'open upskill')
        for verb in ["open", "launch", "start", "close", "quit"]:
            if lower.startswith(verb) and len(lower) > len(verb) and lower[len(verb)] != " ":
                lower = verb + " " + lower[len(verb):]
                break

        if "screenshot" in lower:
            return BoundAction(tool_name="take_screenshot", arguments={}, target="screenshot")
        if any(p in lower for p in ["what is on my screen", "read screen", "read this page"]):
            return BoundAction(tool_name="read_screen", arguments={}, target="screen_text")
        if "search youtube" in lower or "youtube for" in lower:
            query = self._extract_after(lower, ["search youtube for", "search youtube", "youtube for"])
            return BoundAction(tool_name="browser_search", arguments={"query": query or text, "engine": "youtube"}, target=query or text)
        if lower.startswith("search ") or "search google" in lower:
            query = self._extract_after(lower, ["search google for", "search for", "search google", "search"])
            return BoundAction(tool_name="browser_search", arguments={"query": query or text, "engine": "google"}, target=query or text)

        # Syntactic intent disambiguation for unambiguous prefix commands
        if any(lower.startswith(p) for p in ["open ", "launch ", "start "]) and intent in ["other", "switch_window", "terminal_command"]:
            intent = "open_application"
        elif any(lower.startswith(p) for p in ["switch to ", "focus ", "bring up ", "go to "]) and intent in ["other", "open_application"]:
            intent = "switch_window"
        elif any(lower.startswith(p) for p in ["close ", "quit ", "exit ", "close this"]) and intent in ["other", "window_action"]:
            intent = "close_application"
        elif any(lower.startswith(p) for p in ["scroll down", "scroll up", "scroll "]) and intent == "other":
            intent = "scroll"
        elif any(lower.startswith(p) for p in ["minimize", "maximize", "fullscreen"]):
            intent = "window_action"
        elif any(lower.startswith(p) for p in ["type ", "type this"]):
            intent = "input_action"
        elif any(lower.startswith(p) for p in ["press "]):
            intent = "input_action"

        # 1. Open Application
        if intent == "open_application":
            target = self._extract_after(lower, ["open", "launch", "start"])
            if not target:
                target = text

            # Check if it's a folder first
            if self._is_folder_request(target):
                return BoundAction(
                    tool_name="open_application",
                    arguments={"name": target},
                    target=target,
                )

            # Resolve using AppCatalog (exact > descriptor_stripped > alias > phonetic > semantic)
            resolved = self.catalog.resolve(target)
            if resolved:
                if resolved.needs_clarification:
                    return BoundAction(
                        tool_name="clarify",
                        arguments={
                            "prompt": resolved.clarification_prompt,
                            "candidates": resolved.candidates,
                            "target": target,
                        },
                        target=target,
                    )
                return BoundAction(
                    tool_name="open_application",
                    arguments={
                        "name": resolved.name,
                        "desktop_id": resolved.desktop_id,
                        "wm_class": resolved.wm_class,
                        "exec_cmd": resolved.exec_cmd,
                        "candidate_wm_classes": getattr(resolved, "candidate_wm_classes", [resolved.wm_class]),
                    },
                    target=resolved.name,
                )

            # Application resolution failed and not a folder -> Return None (never guess!)
            logger.warning("Application resolution failed for '%s' (not installed or unresolvable). Refusing to guess.", target)
            return None

        # 2. Close Application
        elif intent == "close_application":
            target = self._extract_after(lower, ["close", "quit", "exit"])
            if not target or target in ["this", "this window", "it", "window"]:
                return BoundAction(
                    tool_name="close_application",
                    arguments={},
                    target="active_window",
                )
            resolved = self.catalog.resolve(target)
            if resolved:
                return BoundAction(
                    tool_name="close_application",
                    arguments={"name": resolved.name, "wm_class": resolved.wm_class},
                    target=resolved.name,
                )
            return BoundAction(
                tool_name="close_application",
                arguments={"name": target},
                target=target,
            )

        # 3. Switch Window
        elif intent == "switch_window":
            target = self._extract_after(lower, ["switch to", "switch", "focus", "bring up", "go to"])
            if not target or target in ["previous", "this", "this window", "this app", "this application"]:
                return BoundAction(
                    tool_name="switch_window",
                    arguments={"query": ""},
                    target="previous",
                )

            # If target is a generic category (e.g. "browser", "terminal", "files"), pass directly to SwitchWindowTool
            generic_categories = {"browser", "the browser", "web browser", "terminal", "console", "files", "file manager", "editor", "chat", "notes"}
            if target.lower().strip() in generic_categories:
                return BoundAction(
                    tool_name="switch_window",
                    arguments={"query": target},
                    target=target,
                )

            resolved = self.catalog.resolve(target)
            if resolved:
                return BoundAction(
                    tool_name="switch_window",
                    arguments={"query": resolved.name, "wm_class": resolved.wm_class},
                    target=resolved.name,
                )
            return BoundAction(
                tool_name="switch_window",
                arguments={"query": target},
                target=target,
            )

        # 4. Window Action (minimize, maximize)
        elif intent == "window_action":
            if "maximize" in lower or "fullscreen" in lower:
                return BoundAction(tool_name="maximize_window", arguments={}, target="maximize")
            else:
                return BoundAction(tool_name="minimize_window", arguments={}, target="minimize")

        # 5. Scroll
        elif intent == "scroll":
            direction = "up" if "up" in lower else "down"
            count = 1
            if "twice" in lower or "2 times" in lower or "two times" in lower:
                count = 2
            elif "three times" in lower or "thrice" in lower:
                count = 3
            return BoundAction(
                tool_name="scroll",
                arguments={"direction": direction, "count": count},
                target=direction,
            )

        # 6. Browser Action
        elif intent == "browser_action":
            if "youtube" in lower:
                query = self._extract_after(lower, ["search youtube for", "search youtube", "youtube for", "youtube"])
                return BoundAction(
                    tool_name="browser_search",
                    arguments={"query": query, "engine": "youtube"},
                    target=query,
                )
            elif "search" in lower:
                query = self._extract_after(lower, ["search for", "search google for", "search google", "search"])
                return BoundAction(
                    tool_name="browser_search",
                    arguments={"query": query, "engine": "google"},
                    target=query,
                )
            elif "http" in lower or ".com" in lower or ".org" in lower:
                words = text.split()
                url = next((w for w in words if "." in w), text)
                return BoundAction(
                    tool_name="browser_navigate",
                    arguments={"url": url},
                    target=url,
                )
            else:
                return BoundAction(
                    tool_name="browser_search",
                    arguments={"query": text, "engine": "google"},
                    target=text,
                )

        # 7. Screen Query
        elif intent == "screen_query":
            if "screenshot" in lower:
                return BoundAction(tool_name="take_screenshot", arguments={}, target="screenshot")
            else:
                return BoundAction(tool_name="read_screen", arguments={}, target="screen_text")

        # 8. Input Action
        elif intent == "input_action":
            if "press enter" in lower or "hit enter" in lower:
                return BoundAction(tool_name="press_key", arguments={"key": "Return"}, target="Return")
            elif "press tab" in lower:
                return BoundAction(tool_name="press_key", arguments={"key": "Tab"}, target="Tab")
            elif "press escape" in lower:
                return BoundAction(tool_name="press_key", arguments={"key": "Escape"}, target="Escape")
            else:
                type_content = self._extract_after(lower, ["type this message", "type message", "type"])
                return BoundAction(tool_name="type_text", arguments={"text": type_content or text}, target="typing")

        # 9. System Setting
        elif intent == "system_setting":
            if "mute" in lower:
                return BoundAction(tool_name="mute_volume", arguments={"state": "toggle"}, target="mute")
            elif "volume" in lower:
                match = re.search(r"\d+", lower)
                level = match.group(0) + "%" if match else ("up" if "up" in lower else "down")
                return BoundAction(tool_name="set_volume", arguments={"level": level}, target=str(level))
            elif any(w in lower for w in ["play", "pause", "music", "media"]):
                return BoundAction(tool_name="media_control", arguments={"action": "play_pause"}, target="media")

        return None

    def _extract_after(self, text: str, triggers: list) -> str:
        for trig in sorted(triggers, key=len, reverse=True):
            if trig in text:
                parts = text.split(trig, 1)
                if len(parts) > 1:
                    res = parts[1].strip()
                    # Remove trailing punctuation or filler words
                    res = re.sub(r"^(please|for me|the|a|an|my|our)\s+", "", res, flags=re.IGNORECASE)
                    res = re.sub(r"[.?!,]+$", "", res)
                    return res.strip()
        return ""
