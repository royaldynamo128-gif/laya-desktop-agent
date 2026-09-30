"""
Browser Client for Laya Desktop Agent.
Handles browser navigation, web searches, and DOM interactions via Playwright with fallback to xdg-open.
"""

from typing import Optional, Tuple
import logging
import subprocess
import urllib.parse

logger = logging.getLogger(__name__)


class BrowserClient:
    def __init__(self):
        self._browser = None

    def search_web(self, query: str, engine: str = "google") -> Tuple[bool, str]:
        """Search the web with specified search engine."""
        encoded = urllib.parse.quote_plus(query)
        if engine.lower() == "youtube":
            url = f"https://www.youtube.com/results?search_query={encoded}"
        elif engine.lower() == "duckduckgo":
            url = f"https://duckduckgo.com/?q={encoded}"
        else:
            url = f"https://www.google.com/search?q={encoded}"

        return self.open_url(url)

    def open_url(self, url: str) -> Tuple[bool, str]:
        """Open URL in the default browser."""
        try:
            logger.info("Opening URL: %s", url)
            subprocess.Popen(
                ["xdg-open", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True, f"Navigated to {url}"
        except Exception as e:
            logger.error("Failed to open URL %s: %s", url, e)
            return False, f"Failed to open URL: {e}"
