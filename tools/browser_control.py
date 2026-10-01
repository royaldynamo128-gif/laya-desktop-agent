"""
Browser Control Tools for Laya Desktop Agent.
Handles searching (Google/YouTube/DuckDuckGo), navigation, downloads and
DOM-based element interaction, with honest verification reporting.
"""

from typing import Optional, Dict, Any
import logging
from tools.base import Tool, ToolResult, SafetyTier
from integrations.playwright_client import BrowserClient

logger = logging.getLogger(__name__)


class BrowserSearchTool(Tool):
    name = "browser_search"
    description = "Search the web or YouTube (e.g. 'search YouTube for Linux tutorials')"
    safety_tier = SafetyTier.SAFE

    def __init__(self, browser: BrowserClient):
        self.browser = browser

    def execute(self, query: str, engine: str = "google", **kwargs) -> ToolResult:
        if not query:
            return ToolResult(success=False, message="Search query cannot be empty.", error="Missing query")

        success, msg, info = self.browser.search_web(query, engine=engine)
        return ToolResult(
            success=success,
            message=msg,
            data={"query": query, "engine": engine, **info},
            error=None if success else msg,
            observation_required=success,
        )


class BrowserNavigateTool(Tool):
    name = "browser_navigate"
    description = "Open or navigate to a web URL"
    safety_tier = SafetyTier.SAFE

    def __init__(self, browser: BrowserClient):
        self.browser = browser

    def execute(self, url: str, **kwargs) -> ToolResult:
        if not url:
            return ToolResult(success=False, message="URL cannot be empty.", error="Missing URL")

        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url

        success, msg, info = self.browser.open_url(url)
        return ToolResult(
            success=success,
            message=msg,
            data={"url": url, **info},
            error=None if success else msg,
            observation_required=success,
        )


class BrowserDownloadTool(Tool):
    name = "download_file"
    description = "Download a file from a URL to the Downloads folder"
    safety_tier = SafetyTier.SAFE

    def __init__(self, browser: BrowserClient):
        self.browser = browser

    def execute(self, url: str = "", destination: str = "", what: str = "", **kwargs) -> ToolResult:
        # "download this file" with no URL: resolve the current page's target
        if not url:
            info = self.browser.get_page_info()
            if info and info.get("url"):
                url = info["url"]
            else:
                return ToolResult(
                    success=False,
                    message="No URL to download specified and no active browser page.",
                    error="MissingURL",
                )

        dest = destination or None
        success, msg, path = self.browser.download(url, dest)
        return ToolResult(
            success=success,
            message=msg,
            data={"url": url, "path": path},
            error=None if success else msg,
            observation_required=False,
        )


class BrowserClickTool(Tool):
    """DOM-based element clicking inside the current browser page."""
    name = "browser_click"
    description = "Click a link or button by its text in the current browser page"
    safety_tier = SafetyTier.SAFE

    def __init__(self, browser: BrowserClient):
        self.browser = browser

    def execute(self, element: str = "", text: str = "", **kwargs) -> ToolResult:
        label = element or text
        if not label:
            return ToolResult(success=False, message="Element text required.", error="MissingText")
        ok, msg = self.browser.click_text(label)
        return ToolResult(
            success=ok,
            message=msg,
            data={"element": label},
            error=None if ok else msg,
            observation_required=ok,
        )


class BrowserFillTool(Tool):
    """Fill a form field in the current browser page by placeholder/label."""
    name = "browser_fill"
    description = "Fill a text field in the current browser page"
    safety_tier = SafetyTier.SAFE

    def __init__(self, browser: BrowserClient):
        self.browser = browser

    def execute(self, field: str = "", value: str = "", **kwargs) -> ToolResult:
        if value is None or str(value) == "":
            return ToolResult(success=False, message="Value required.", error="MissingValue")
        ok, msg = self.browser.fill_field(field, str(value))
        return ToolResult(
            success=ok,
            message=msg,
            data={"field": field, "value": str(value)},
            error=None if ok else msg,
            observation_required=ok,
        )
