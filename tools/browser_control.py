"""
Browser Control Tools for Laya Desktop Agent.
Handles searching YouTube, Google, and opening web URLs.
"""

from typing import Optional
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

        success, msg = self.browser.search_web(query, engine=engine)
        return ToolResult(
            success=success,
            message=f"Searching {engine} for '{query}'." if success else msg,
            data={"query": query, "engine": engine},
            error=None if success else msg,
            observation_required=True,
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

        success, msg = self.browser.open_url(url)
        return ToolResult(
            success=success,
            message=f"Navigating to {url}." if success else msg,
            data={"url": url},
            error=None if success else msg,
            observation_required=True,
        )
