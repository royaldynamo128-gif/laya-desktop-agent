"""
Parameter Binder for Laya Desktop Agent.

Binds parsed semantic clauses (from planning.semantic_parser) to concrete tool
calls. Entity resolution (application names, folders, files) is delegated to
the dynamic AppCatalog and FileResolver — never hardcoded command strings.

Back-compat: bind(transcription, decision) still works and now routes
through the semantic parser instead of regex chains.
"""

from typing import Dict, Any, Optional, Tuple, List
import logging
import re

from models.laya_engine import LayaDecision
from integrations.app_catalog import AppCatalog
from planning.semantic_parser import Clause, SemanticParser, get_parser
from core.action_types import BoundAction

logger = logging.getLogger(__name__)


# Map parser intents to executor tool names
INTENT_TOOL_MAP = {
    "open_application": "open_application",
    "close_application": "close_application",
    "switch_window": "switch_window",
    "window_action": "window_action",       # handled specially below
    "scroll": "scroll",
    "browser_search": "browser_search",
    "browser_navigate": "browser_navigate",
    "type_text": "type_text",
    "press_key": "press_key",
    "take_screenshot": "take_screenshot",
    "read_screen": "read_screen",
    "set_volume": "set_volume",
    "mute_volume": "mute_volume",
    "media_control": "media_control",
    "click_element": "click_element",
    "submit_action": "submit_action",
    "fill_form": "fill_form",
    "ui_action": "ui_action",
    "download_action": "download_file",
    "save_action": "save_as",
    "file_open": "file_open",
    "file_move": "file_move",
    "file_copy": "file_copy",
    "file_rename": "file_rename",
    "file_delete": "file_delete",
    "file_create_folder": "file_create_folder",
    "file_create_file": "file_create_folder",
    "file_compress": "file_compress",
    "file_extract": "file_extract",
    "file_find": "file_find",
    "wait_action": "wait_action",
    "verify_action": "verify_action",
    "input_action": "type_text",
}


class ParameterBinder:
    def __init__(self, catalog: Optional[AppCatalog] = None):
        self.catalog = catalog or AppCatalog()
        self.parser: SemanticParser = get_parser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bind(self, transcription: str, decision: Optional[LayaDecision] = None) -> Optional[BoundAction]:
        """Map a raw command (with optional Laya decision hint) to a tool action."""
        if not transcription or not transcription.strip():
            return None

        result = self.parser.parse(transcription)
        if not result.clauses:
            return None

        # Laya intent hint can disambiguate unknown clauses
        clause = result.clauses[0]
        hint = decision.intent if decision else None
        return self.bind_clause(clause, hint=hint, context_text=transcription)

    def bind_clause(
        self,
        clause: Clause,
        hint: Optional[str] = None,
        context_text: str = "",
    ) -> Optional[BoundAction]:
        """Bind one parsed clause to a concrete tool action."""
        intent = clause.intent
        if intent == "unknown" and hint and hint in INTENT_TOOL_MAP:
            # Router hint disambiguation
            clause = self._reparse_with_hint(clause, hint)
            intent = clause.intent

        tool = INTENT_TOOL_MAP.get(intent)
        if not tool:
            logger.info("No tool mapping for clause intent '%s' (raw: '%s')", intent, clause.raw)
            return None

        # Intents with a specific binder: a None result is FINAL (it means the
        # specific binder refused to bind — e.g. unresolvable application —
        # and we must NEVER fall through to a generic guess).
        intents_with_final_binders = {
            "open_application", "close_application", "switch_window", "window_action",
            "browser_search", "browser_navigate", "type_text", "press_key",
            "click_element", "submit_action", "ui_action", "download_action",
            "save_action", "fill_form", "wait_action", "verify_action",
            "file_open", "file_move", "file_copy", "file_rename", "file_delete",
            "file_create_folder", "file_compress", "file_extract", "file_find",
            "scroll", "set_volume", "mute_volume", "media_control",
            "take_screenshot", "read_screen",
        }

        binder_method = getattr(self, f"_bind_{intent}", None)
        if binder_method:
            action = binder_method(clause)
            if action is not None:
                return action
            if intent in intents_with_final_binders:
                return None
            # (no generic fallback for these intents — None is a refusal)

        # Generic pass-through for intents without a specific binder
        args = dict(clause.arguments)
        if clause.is_referring and "last_referenced" not in args:
            args["_referring"] = clause.target
        return BoundAction(tool_name=tool, arguments=args, target=clause.target)

    # ------------------------------------------------------------------
    # Intent-specific binders
    # ------------------------------------------------------------------

    def _bind_open_application(self, clause: Clause) -> Optional[BoundAction]:
        args = clause.arguments
        if args.get("is_folder"):
            return BoundAction(
                tool_name="open_application",
                arguments={"name": clause.target, "folder": args.get("folder", clause.target)},
                target=clause.target,
            )

        target = args.get("name") or clause.target
        if not target:
            return None

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

        # Unresolvable: never guess — but plural referring phrases ("these two
        # applications", "those apps") are a request for clarification, not a
        # resolution failure.
        if clause.is_referring or re.search(
            r"\b(these|those|two|three|four|five|\d+)\b[\w\s]*\b(app|application|program|window)s?\b",
            target, flags=re.IGNORECASE,
        ):
            logger.info("Plural/underspecified application reference '%s' → asking for clarification.", target)
            return BoundAction(
                tool_name="clarify",
                arguments={"prompt": f"Which applications would you like me to open?"},
                target=target,
            )
        logger.warning("Application resolution failed for '%s' (not installed or unresolvable).", target)
        return None

    def _bind_close_application(self, clause: Clause) -> BoundAction:
        target = clause.arguments.get("name") or clause.target
        if not target or target == "active_window":
            return BoundAction(tool_name="close_application", arguments={}, target="active_window")
        resolved = self.catalog.resolve(target)
        if resolved:
            return BoundAction(
                tool_name="close_application",
                arguments={"name": resolved.name, "wm_class": resolved.wm_class,
                           "candidate_wm_classes": getattr(resolved, "candidate_wm_classes", [resolved.wm_class])},
                target=resolved.name,
            )
        return BoundAction(tool_name="close_application", arguments={"name": target}, target=target)

    # Generic window categories that SwitchWindowTool resolves itself via
    # CATEGORY_WINDOW_CLASSES (browser/terminal/files/editor...)
    GENERIC_SWITCH_CATEGORIES = {
        "browser", "the browser", "web browser", "default browser", "my browser",
        "internet", "terminal", "my terminal", "console", "files", "file manager",
        "my files", "explorer", "file browser", "editor", "code editor", "text editor",
        "chat", "notes", "music", "media player", "video player", "settings",
    }

    def _bind_switch_window(self, clause: Clause) -> BoundAction:
        target = clause.arguments.get("query") or clause.target
        target_clean = target.lower().strip() if target else ""
        if target_clean in self.GENERIC_SWITCH_CATEGORIES:
            # Pass the category directly; the tool maps it to window classes
            return BoundAction(tool_name="switch_window", arguments={"query": target_clean}, target=target_clean)
        resolved = self.catalog.resolve(target) if target else None
        if resolved:
            return BoundAction(
                tool_name="switch_window",
                arguments={"query": resolved.name, "wm_class": resolved.wm_class,
                           "candidate_wm_classes": getattr(resolved, "candidate_wm_classes", [resolved.wm_class])},
                target=resolved.name,
            )
        return BoundAction(tool_name="switch_window", arguments={"query": target or ""}, target=target or "previous")

    def _bind_window_action(self, clause: Clause) -> BoundAction:
        action = clause.arguments.get("action", "minimize")
        tool = {
            "minimize": "minimize_window",
            "maximize": "maximize_window",
            "fullscreen": "maximize_window",
            "restore": "minimize_window",  # restore ≈ unminimize handled by activate
            "tile": "maximize_window",
        }.get(action, "minimize_window")
        return BoundAction(tool_name=tool, arguments={}, target=action)

    def _bind_browser_search(self, clause: Clause) -> BoundAction:
        query = clause.arguments.get("query") or clause.target
        engine = clause.arguments.get("engine", "google")
        return BoundAction(
            tool_name="browser_search",
            arguments={"query": query, "engine": engine},
            target=query,
        )

    def _bind_browser_navigate(self, clause: Clause) -> BoundAction:
        url = clause.arguments.get("url") or clause.target
        return BoundAction(tool_name="browser_navigate", arguments={"url": url}, target=url)

    def _bind_type_text(self, clause: Clause) -> BoundAction:
        text = clause.arguments.get("text") or clause.target
        return BoundAction(tool_name="type_text", arguments={"text": text}, target=text[:40])

    def _bind_press_key(self, clause: Clause) -> BoundAction:
        key = clause.arguments.get("key") or clause.target
        return BoundAction(tool_name="press_key", arguments={"key": key}, target=key)

    def _bind_click_element(self, clause: Clause) -> BoundAction:
        element = clause.arguments.get("element") or clause.target
        args = {"element": element}
        if clause.arguments.get("ordinal"):
            args["ordinal"] = clause.arguments["ordinal"]
        return BoundAction(tool_name="click_element", arguments=args, target=element)

    def _bind_submit_action(self, clause: Clause) -> BoundAction:
        action = clause.arguments.get("action", "submit")
        return BoundAction(tool_name="submit_action", arguments={"action": action}, target=action)

    def _bind_ui_action(self, clause: Clause) -> BoundAction:
        return BoundAction(
            tool_name="ui_action",
            arguments={"action": clause.arguments.get("action", ""), "object": clause.arguments.get("object", clause.target)},
            target=clause.target,
        )

    def _bind_download_action(self, clause: Clause) -> BoundAction:
        what = clause.arguments.get("what") or clause.target
        args = {"what": what}
        if clause.is_referring:
            args["_referring"] = what
        return BoundAction(tool_name="download_file", arguments=args, target=what)

    def _bind_save_action(self, clause: Clause) -> BoundAction:
        return BoundAction(
            tool_name="save_as",
            arguments={
                "format": clause.arguments.get("format"),
                "location": clause.arguments.get("location"),
            },
            target=clause.arguments.get("format") or "file",
        )

    def _bind_fill_form(self, clause: Clause) -> BoundAction:
        return BoundAction(
            tool_name="fill_form",
            arguments={"form": clause.arguments.get("form", clause.target)},
            target=clause.target,
        )

    def _bind_wait_action(self, clause: Clause) -> BoundAction:
        return BoundAction(
            tool_name="wait_action",
            arguments={
                "for_what": clause.arguments.get("for", clause.target),
                "timeout_sec": clause.arguments.get("timeout_sec", 30.0),
            },
            target=clause.target,
        )

    def _bind_verify_action(self, clause: Clause) -> BoundAction:
        return BoundAction(
            tool_name="verify_action",
            arguments={"expectation": clause.arguments.get("expectation", clause.target)},
            target=clause.target,
        )

    def _bind_file_open(self, clause: Clause) -> BoundAction:
        return BoundAction(
            tool_name="file_open",
            arguments={"target": clause.arguments.get("target", clause.target)},
            target=clause.target,
        )

    def _bind_file_move(self, clause: Clause) -> BoundAction:
        args = {
            "source": clause.arguments.get("source", ""),
            "destination": clause.arguments.get("destination", ""),
        }
        return BoundAction(tool_name="file_move", arguments=args, target=args["source"])

    def _bind_file_copy(self, clause: Clause) -> BoundAction:
        args = {
            "source": clause.arguments.get("source", ""),
            "destination": clause.arguments.get("destination", ""),
        }
        return BoundAction(tool_name="file_copy", arguments=args, target=args["source"])

    def _bind_file_rename(self, clause: Clause) -> BoundAction:
        args = {
            "source": clause.arguments.get("source", ""),
            "new_name": clause.arguments.get("new_name", ""),
        }
        if clause.quoted_payload and not args["new_name"]:
            args["new_name"] = clause.quoted_payload
        return BoundAction(tool_name="file_rename", arguments=args, target=args["source"])

    def _bind_file_delete(self, clause: Clause) -> BoundAction:
        return BoundAction(
            tool_name="file_delete",
            arguments={"target": clause.arguments.get("target", clause.target)},
            target=clause.target,
        )

    def _bind_file_create_folder(self, clause: Clause) -> BoundAction:
        return BoundAction(
            tool_name="file_create_folder",
            arguments={"name": clause.arguments.get("name", clause.target)},
            target=clause.target,
        )

    def _bind_file_compress(self, clause: Clause) -> BoundAction:
        return BoundAction(
            tool_name="file_compress",
            arguments={"targets": clause.arguments.get("targets", clause.target)},
            target=clause.target,
        )

    def _bind_file_extract(self, clause: Clause) -> BoundAction:
        return BoundAction(
            tool_name="file_extract",
            arguments={"target": clause.arguments.get("target", clause.target)},
            target=clause.target,
        )

    def _bind_file_find(self, clause: Clause) -> BoundAction:
        return BoundAction(
            tool_name="file_find",
            arguments={
                "criterion": clause.arguments.get("criterion", "largest"),
                "pattern": clause.arguments.get("pattern", ""),
            },
            target=clause.target,
        )

    def _bind_scroll(self, clause: Clause) -> BoundAction:
        return BoundAction(
            tool_name="scroll",
            arguments={
                "direction": clause.arguments.get("direction", "down"),
                "count": clause.arguments.get("count", 1),
            },
            target=clause.arguments.get("direction", "down"),
        )

    def _bind_set_volume(self, clause: Clause) -> BoundAction:
        level = clause.arguments.get("level", "50%")
        return BoundAction(tool_name="set_volume", arguments={"level": level}, target=str(level))

    def _bind_mute_volume(self, clause: Clause) -> BoundAction:
        state = clause.arguments.get("state", "toggle")
        return BoundAction(tool_name="mute_volume", arguments={"state": state}, target=state)

    def _bind_media_control(self, clause: Clause) -> BoundAction:
        action = clause.arguments.get("action", "play_pause")
        return BoundAction(tool_name="media_control", arguments={"action": action}, target=action)

    def _bind_take_screenshot(self, clause: Clause) -> BoundAction:
        return BoundAction(tool_name="take_screenshot", arguments={}, target="screenshot")

    def _bind_read_screen(self, clause: Clause) -> BoundAction:
        return BoundAction(tool_name="read_screen", arguments={}, target="screen_text")

    def _bind_type_or_input(self, clause: Clause) -> BoundAction:
        return self._bind_type_text(clause)

    # ------------------------------------------------------------------
    # Hint-based reparse: if Laya router says the intent is X but the clause
    # was unknown, reparse treating the first word as that intent's verb.
    # ------------------------------------------------------------------

    def _reparse_with_hint(self, clause: Clause, hint: str) -> Clause:
        verb_map = {
            "open_application": "open",
            "close_application": "close",
            "switch_window": "switch to",
            "browser_action": "search",
            "input_action": "type",
            "screen_query": "read",
            "system_setting": "set volume",
            "window_action": "minimize",
            "scroll": "scroll",
        }
        verb = verb_map.get(hint)
        if not verb:
            return clause
        reparse_text = f"{verb} {clause.raw}" if not clause.raw.lower().startswith(verb) else clause.raw
        reparsed = self.parser.parse(reparse_text)
        if reparsed.clauses and reparsed.clauses[0].intent != "unknown":
            return reparsed.clauses[0]
        return clause
