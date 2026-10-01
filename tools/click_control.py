"""
UI Interaction Tools for Laya Desktop Agent.

Click, select, submit, fill and generic UI actions resolved through the
ScreenAnalyzer perception layer (AT-SPI first, OCR vision fallback, keyboard
navigation last). Never hardcoded coordinates: elements are located by their
semantic label in the CURRENT screen state.
"""

from typing import Any, Dict, List, Optional
import logging
import re
import time

from tools.base import Tool, ToolResult, SafetyTier
from vision.screen_analyzer import ScreenAnalyzer

logger = logging.getLogger(__name__)


class ClickElementTool(Tool):
    name = "click_element"
    description = "Click / activate a UI element by its visible label (e.g. 'click the Proceed button', 'select 5x')"
    safety_tier = SafetyTier.SAFE

    ORDINALS = {"first": 1, "top": 1, "latest": 1, "newest": 1, "second": 2, "third": 3, "next": 1}

    RESULT_LABEL_RE = re.compile(
        r"^(search result|results?|(first|top|second|third|latest|newest) result|video result|result link)$",
        re.IGNORECASE,
    )

    def __init__(self, analyzer: ScreenAnalyzer, keyboard_tool=None, browser_client=None):
        self.analyzer = analyzer
        self.keyboard = keyboard_tool  # PressKeyTool fallback
        self.browser = browser_client  # optional BrowserClient (DOM tier)

    def execute(
        self,
        element: str = "",
        label: str = "",
        ordinal: str = "",
        query: str = "",
        wait_sec: float = 0.0,
        retries: int = 2,
        **kwargs,
    ) -> ToolResult:
        target_label = (element or label or "").strip()
        if not target_label:
            return ToolResult(success=False, message="Element label required.", error="MissingLabel")

        if wait_sec > 0:
            time.sleep(min(wait_sec, 5.0))

        # Search-result references: DOM control first (precise on real
        # browsers), then on-screen link-role elements (works without DOM).
        if self.RESULT_LABEL_RE.match(target_label):
            idx = self.ORDINALS.get((ordinal or "first").lower(), 1)
            return self._click_search_result(idx, retries, query=(query or "").strip())

        last_reason = "element not found"

        for attempt in range(1, max(1, retries) + 1):
            loc = self.analyzer.find_element(target_label)
            if not loc:
                last_reason = f"No UI element matching '{target_label}' found on screen"
                logger.info("ClickElement attempt %d: %s", attempt, last_reason)
                if attempt < retries:
                    time.sleep(0.4)  # UI may still be rendering
                continue

            ok = self.analyzer.activate(loc)
            if ok:
                return ToolResult(
                    success=True,
                    message=f"Activated '{loc.label}' via {loc.method}.",
                    data={
                        "element": target_label,
                        "matched_label": loc.label,
                        "method": loc.method,
                        "role": loc.role,
                        "x": loc.x, "y": loc.y,
                    },
                    observation_required=True,
                )
            last_reason = f"Found '{loc.label}' but activation failed"

        # Keyboard navigation fallback: Tab-cycle is unreliable; report honestly
        return ToolResult(
            success=False,
            message=f"Could not activate '{target_label}': {last_reason}.",
            error="ElementActivationFailed",
            data={"element": target_label},
        )

    def _click_search_result(self, index: int, retries: int, query: str = "") -> ToolResult:
        last_reason = "no mechanism located a result"

        for attempt in range(1, max(1, retries) + 1):
            # Tier 1: browser DOM (engine-aware result selectors)
            if self.browser is not None:
                try:
                    if self.browser.has_dom_control:
                        ok, msg = self.browser.click_first_result(index)
                        if ok:
                            return ToolResult(
                                success=True,
                                message=msg,
                                data={"element": "search result", "index": index, "method": "dom", "query": query},
                                observation_required=True,
                            )
                        last_reason = msg
                except Exception as e:
                    logger.debug("DOM result click failed: %s", e)
                    last_reason = f"DOM click failed: {e}"

            # Tier 2: on-screen link-role elements (AT-SPI → coordinates)
            finder = getattr(self.analyzer, "_find_result_link", None)
            if callable(finder):
                loc = finder(query=query)
                if loc:
                    ok = self.analyzer.activate(loc)
                    if ok:
                        return ToolResult(
                            success=True,
                            message=f"Opened search result '{loc.label}' via {loc.method}.",
                            data={
                                "element": "search result", "index": index, "query": query,
                                "matched_label": loc.label, "method": loc.method,
                                "role": loc.role, "x": loc.x, "y": loc.y,
                            },
                            observation_required=True,
                        )
                    last_reason = f"Found result link '{loc.label}' but activation failed"

            if attempt < retries:
                time.sleep(0.5)  # results page may still be rendering

        return ToolResult(
            success=False,
            message=f"Could not open search result #{index}: {last_reason}.",
            error="SearchResultNotFound",
            data={"element": "search result", "index": index, "query": query},
        )


class SelectOptionTool(Tool):
    name = "select_option"
    description = "Select an option value in the active window (e.g. 'select 5x')"
    safety_tier = SafetyTier.SAFE

    def __init__(self, analyzer: ScreenAnalyzer, click_tool: Optional[ClickElementTool] = None):
        self.analyzer = analyzer
        self.click_tool = click_tool

    def execute(self, value: str = "", option: str = "", **kwargs) -> ToolResult:
        val = (value or option or "").strip()
        if not val:
            return ToolResult(success=False, message="Option value required.", error="MissingValue")
        click = self.click_tool or ClickElementTool(self.analyzer)
        return click.execute(element=val)


class SubmitTool(Tool):
    name = "submit_action"
    description = "Activate the primary submit/proceed/confirm control in the active window"
    safety_tier = SafetyTier.SAFE

    SUBMIT_LABELS = [
        "submit", "proceed", "continue", "confirm", "ok", "next", "finish",
        "done", "apply", "save", "send", "go", "yes", "accept", "download",
    ]

    def __init__(self, analyzer: ScreenAnalyzer, click_tool: Optional[ClickElementTool] = None, press_key_tool=None):
        self.analyzer = analyzer
        self.click_tool = click_tool
        self.press = press_key_tool

    def execute(self, action: str = "submit", **kwargs) -> ToolResult:
        verb = (action or "submit").strip().lower()

        # Try the specific verb first, then generic submit labels
        labels = [verb] + [l for l in self.SUBMIT_LABELS if l != verb]
        click = self.click_tool or ClickElementTool(self.analyzer)

        for label in labels:
            loc = self.analyzer.find_element(label)
            if loc:
                ok = self.analyzer.activate(loc)
                if ok:
                    return ToolResult(
                        success=True,
                        message=f"Activated '{loc.label}' (submit action '{verb}').",
                        data={"element": loc.label, "method": loc.method},
                        observation_required=True,
                    )

        # Keyboard fallback: Enter often submits forms/dialogs
        if self.press:
            res = self.press.execute(key="Return")
            if res.success:
                return ToolResult(
                    success=True,
                    message=f"Submitted via Enter key (no visible submit control found).",
                    data={"element": "Return", "method": "keyboard"},
                    observation_required=True,
                )

        return ToolResult(
            success=False,
            message=f"No submit control found for '{verb}'.",
            error="SubmitControlNotFound",
        )


class FillFormTool(Tool):
    name = "fill_form"
    description = "Fill text fields in the active window/form using provided values"
    safety_tier = SafetyTier.SAFE

    def __init__(self, analyzer: ScreenAnalyzer, type_tool=None, press_key_tool=None):
        self.analyzer = analyzer
        self.type_tool = type_tool
        self.press = press_key_tool

    def execute(self, form: str = "", fields: Optional[Dict[str, str]] = None, values: Optional[List[str]] = None, **kwargs) -> ToolResult:
        """Fill form fields. values = ordered list of text to type into each field."""
        entries = values or []
        if fields:
            entries = list(fields.values())

        atspi = getattr(self.analyzer, "atspi", None)
        editable = []
        if atspi and getattr(atspi, "enabled", False):
            try:
                editable = atspi.get_editable_fields(max_fields=len(entries) or 6)
            except Exception as e:
                logger.debug("get_editable_fields failed: %s", e)

        if not editable:
            return ToolResult(
                success=False,
                message="No editable form fields found in the active window.",
                error="NoFieldsFound",
            )

        filled = 0
        for i, value in enumerate(entries):
            if i >= len(editable):
                break
            field = editable[i]
            acc = field.get("accessible")
            try:
                # Focus the field then type via keyboard synthesis
                comp = acc.get_component_iface() if acc else None
                if comp and comp.grab_focus():
                    if self.type_tool:
                        r = self.type_tool.execute(text=str(value))
                        if r.success:
                            filled += 1
                            if self.press:
                                self.press.execute(key="Tab")
                            continue
                # Direct text interface insertion
                txt = acc.get_text_iface() if acc else None
                if txt and hasattr(txt, "set_text"):
                    txt.set_text(str(value))
                    filled += 1
            except Exception as e:
                logger.debug("Field fill failed: %s", e)

        if filled:
            return ToolResult(
                success=True,
                message=f"Filled {filled} form field(s).",
                data={"filled": filled, "total_fields": len(editable)},
                observation_required=True,
            )
        return ToolResult(
            success=False,
            message="Could not fill any form fields.",
            error="FillFailed",
        )


class UIInteractTool(Tool):
    """
    Generic UI interaction for arbitrary verbs (add/import/resize/...).
    Resolves the verb+object phrase to a matching on-screen control.
    E.g. "add this image" → an "Add Image" button or an "Import" menu entry.
    """
    name = "ui_action"
    description = "Perform an app UI action by matching its label on screen (add/import/apply/resize...)"
    safety_tier = SafetyTier.SAFE

    ACTION_LABEL_VARIANTS = {
        "add": ["add", "add image", "add file", "open image", "import", "load", "upload", "browse"],
        "import": ["import", "import image", "open file", "add", "load", "browse", "upload"],
        "insert": ["insert", "add", "import"],
        "upload": ["upload", "add", "import", "browse"],
        "attach": ["attach", "add", "upload"],
        "open": ["open", "open file", "browse", "play", "install", "launch", "show"],
        "resize": ["resize", "image size", "scale", "resize image"],
        "apply": ["apply", "ok", "confirm"],
        "adjust": ["adjust", "settings", "options"],
    }

    def __init__(self, analyzer: ScreenAnalyzer, click_tool: Optional[ClickElementTool] = None):
        self.analyzer = analyzer
        self.click_tool = click_tool or ClickElementTool(analyzer)

    def execute(self, action: str = "", object_name: str = "", object: str = "", **kwargs) -> ToolResult:
        act = (action or "").strip().lower()
        obj = (object_name or object or "").strip()

        # Build candidate label list: verb-object combos, then verb alone
        candidates: List[str] = []
        obj_clean = obj
        for prefix in ("this ", "the ", "a ", "my "):
            if obj_clean.lower().startswith(prefix):
                obj_clean = obj_clean[len(prefix):]
                break

        if act and obj_clean:
            candidates.extend([
                f"{act} {obj_clean}".title(),
                f"{act} {obj_clean}",
                obj_clean.title(),
                obj_clean,
            ])
        # Common app-level synonyms for the verb
        for v in self.ACTION_LABEL_VARIANTS.get(act, []):
            candidates.append(v.title())
            if obj_clean:
                candidates.append(f"{v} {obj_clean}".title())
                candidates.append(f"{v} {obj_clean}")

        # De-duplicate preserving order
        seen = set()
        uniq = []
        for c in candidates:
            cl = c.lower()
            if cl not in seen:
                seen.add(cl)
                uniq.append(c)

        for label in uniq:
            loc = self.analyzer.find_element(label)
            if loc:
                ok = self.analyzer.activate(loc)
                if ok:
                    return ToolResult(
                        success=True,
                        message=f"Performed '{act or 'action'}' via '{loc.label}' ({loc.method}).",
                        data={"action": act, "object": obj_clean, "element": loc.label, "method": loc.method},
                        observation_required=True,
                    )

        return ToolResult(
            success=False,
            message=f"No on-screen control found for '{act} {obj_clean}'.",
            error="UIActionTargetNotFound",
            data={"action": act, "object": obj_clean, "tried_labels": uniq[:8]},
        )


class WaitStateTool(Tool):
    name = "wait_action"
    description = "Wait until a condition is met (element visible / file downloaded) with timeout"
    safety_tier = SafetyTier.SAFE

    def __init__(self, analyzer: Optional[ScreenAnalyzer] = None):
        self.analyzer = analyzer

    def execute(self, for_what: str = "", expectation: str = "", timeout_sec: float = 30.0, poll_sec: float = 0.5, **kwargs) -> ToolResult:
        what = (for_what or expectation or "").strip().lower()
        if not what:
            time.sleep(min(timeout_sec, 1.0))
            return ToolResult(success=True, message="Waited.", data={"waited": True}, observation_required=False)

        deadline = time.time() + min(timeout_sec, 120.0)

        if "download" in what:
            # Poll the Downloads folder for a new file
            from pathlib import Path
            from tools.file_control import get_file_resolver
            dl = Path.home() / "Downloads"
            baseline = {p.name for p in dl.glob("*")} if dl.exists() else set()
            while time.time() < deadline:
                time.sleep(poll_sec)
                current = {p.name for p in dl.glob("*")} if dl.exists() else set()
                new_files = current - baseline
                # Ignore .crdownload/.part partial files
                complete = [f for f in new_files if not f.endswith((".crdownload", ".part", ".tmp"))]
                if complete:
                    newest = max((dl / f for f in complete), key=lambda p: p.stat().st_mtime)
                    return ToolResult(
                        success=True,
                        message=f"Download complete: {newest.name}.",
                        data={"downloaded_file": str(newest)},
                        observation_required=False,
                    )
            return ToolResult(
                success=False,
                message="Timed out waiting for the download.",
                error="WaitTimeout",
                data={"waited_for": what},
            )

        # Element visibility wait
        if self.analyzer:
            label = re.sub(r"^(for|until)\s+", "", what)
            while time.time() < deadline:
                visible, _ = self.analyzer.element_visible(label)
                if visible:
                    return ToolResult(
                        success=True,
                        message=f"'{label}' is now visible.",
                        data={"element": label},
                        observation_required=False,
                    )
                time.sleep(poll_sec)
            return ToolResult(
                success=False,
                message=f"Timed out waiting for '{label}'.",
                error="WaitTimeout",
            )

        time.sleep(min(2.0, timeout_sec))
        return ToolResult(success=True, message="Waited.", observation_required=False)


class VerifyStateTool(Tool):
    name = "verify_action"
    description = "Verify that an expected state is present on screen (element/text visible)"
    safety_tier = SafetyTier.SAFE

    def __init__(self, analyzer: Optional[ScreenAnalyzer] = None):
        self.analyzer = analyzer

    def execute(self, expectation: str = "", **kwargs) -> ToolResult:
        what = (expectation or "").strip()
        if not what:
            return ToolResult(success=False, message="Nothing to verify specified.", error="MissingExpectation")

        if self.analyzer:
            visible, detail = self.analyzer.element_visible(what)
            if visible:
                return ToolResult(
                    success=True,
                    message=f"Verified: {detail}.",
                    data={"expectation": what, "visible": True},
                    observation_required=False,
                )
            return ToolResult(
                success=False,
                message=f"Verification failed: {detail}.",
                error="ExpectationNotMet",
                data={"expectation": what, "visible": False},
            )

        return ToolResult(success=False, message="No perception layer available to verify.", error="NoAnalyzer")

