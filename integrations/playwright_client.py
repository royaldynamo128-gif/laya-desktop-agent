"""
Browser Client for Laya Desktop Agent.

Multi-tier browser automation:
  Tier 1: Playwright over CDP — attach to a running Chromium/Chrome/Opera
          instance (launched with --remote-debugging-port) for full DOM
          control: navigate, search, click by text, fill forms, downloads.
  Tier 2: Playwright-managed headless/headed Chromium instance.
  Tier 3: xdg-open default browser (fire-and-forget; verification falls back
          to window-title inspection — honestly reported as unverified if the
          title cannot be confirmed).

Every tier reports what it actually did; no fake success states.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

ENGINES = {
    "google": "https://www.google.com/search?q={}",
    "youtube": "https://www.youtube.com/results?search_query={}",
    "duckduckgo": "https://duckduckgo.com/?q={}",
    "bing": "https://www.bing.com/search?q={}",
}


class BrowserClient:
    def __init__(self, cdp_port: int = 9222, playwright_mode: str = "auto"):
        self.cdp_port = cdp_port
        self.playwright_mode = playwright_mode
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    # ------------------------------------------------------------------
    # Playwright management
    # ------------------------------------------------------------------

    def _try_connect(self) -> bool:
        """Connect Playwright to a browser: existing CDP endpoint first."""
        if self._page is not None:
            return True
        if self.playwright_mode == "xdg-open":
            return False

        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            logger.debug("Playwright not installed; browser automation falls back to xdg-open")
            return False

        try:
            self._playwright = sync_playwright().start()
        except Exception as e:
            logger.debug("Playwright failed to start: %s", e)
            return False

        # 1. Attach to a running browser via CDP
        try:
            self._browser = self._playwright.chromium.connect_over_cdp(
                f"http://127.0.0.1:{self.cdp_port}", timeout=1500
            )
            contexts = self._browser.contexts
            self._context = contexts[0] if contexts else self._browser.new_context()
            pages = self._context.pages
            self._page = pages[0] if pages else self._context.new_page()
            logger.info("Attached Playwright to running browser via CDP port %d", self.cdp_port)
            return True
        except Exception as e:
            logger.debug("CDP attach failed (no browser running with debug port): %s", e)

        # 2. Launch our own Chromium instance
        try:
            self._context = self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(Path.home() / ".cache" / "laya" / "browser-profile"),
                headless=False,
                args=["--start-maximized"],
            )
            pages = self._context.pages
            self._page = pages[0] if pages else self._context.new_page()
            logger.info("Playwright launched managed Chromium instance")
            return True
        except Exception as e:
            logger.debug("Playwright launch failed: %s", e)
            try:
                self._playwright.stop()
            except Exception:
                pass
            self._playwright = None
            return False

    def _disconnect(self) -> None:
        try:
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    @property
    def has_dom_control(self) -> bool:
        return self._try_connect()

    # ------------------------------------------------------------------
    # High-level operations
    # ------------------------------------------------------------------

    def search_web(self, query: str, engine: str = "google") -> Tuple[bool, str, Dict[str, Any]]:
        """Search the web; returns (success, message, info)."""
        encoded = urllib.parse.quote_plus(query)
        tmpl = ENGINES.get(engine.lower(), ENGINES["google"])
        url = tmpl.format(encoded)
        return self.open_url(url, context={"query": query, "engine": engine})

    def open_url(self, url: str, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """Open a URL. Returns (success, message, info with verification data)."""
        if not url:
            return False, "URL cannot be empty.", {}

        # Tier 1/2: Playwright DOM control
        if self._try_connect():
            try:
                self._page.goto(url, timeout=15000, wait_until="domcontentloaded")
                final_url = self._page.url
                title = self._page.title()
                return True, f"Navigated to {final_url}.", {
                    "url": final_url, "title": title, "method": "playwright",
                    "verified": True, "query": (context or {}).get("query", ""),
                }
            except Exception as e:
                logger.warning("Playwright navigation failed (%s); falling back to xdg-open", e)
                self._disconnect()

        # Tier 3: xdg-open
        opener = shutil.which("xdg-open") or os.environ.get("XDG_OPEN_BIN")
        if opener:
            try:
                subprocess.Popen(
                    [opener, url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                return True, f"Navigated to {url}.", {
                    "url": url, "method": "xdg-open",
                    "verified": False,  # honestly unverified — verifier will check window title
                    "query": (context or {}).get("query", ""),
                }
            except Exception as e:
                return False, f"Failed to open URL: {e}", {"url": url, "error": str(e)}

        return False, "No browser automation mechanism available (no Playwright, no xdg-open).", {"url": url}

    def get_page_info(self) -> Optional[Dict[str, Any]]:
        """Current page URL/title when DOM control is available."""
        if not self._try_connect():
            return None
        try:
            return {"url": self._page.url, "title": self._page.title()}
        except Exception:
            return None

    def click_text(self, text: str) -> Tuple[bool, str]:
        """Click an element by its visible text in the current page (DOM)."""
        if not self._try_connect():
            return False, "No DOM control available."
        try:
            # Prefer links, buttons, then any element
            for selector in ["a", "button", "[role=button]", "input[type=submit]", "div", "span", "li"]:
                try:
                    loc = self._page.locator(selector, has_text=re.compile(re.escape(text), re.IGNORECASE)).first
                    if loc.count() > 0:
                        loc.click(timeout=3000)
                        return True, f"Clicked '{text}' via DOM."
                except Exception:
                    continue
            return False, f"No element matching '{text}' found in page."
        except Exception as e:
            return False, f"DOM click failed: {e}"

    # Engine-aware search-result selectors (index is 1-based)
    RESULT_SELECTORS = [
        # YouTube search results
        "ytd-video-renderer a#thumbnail",
        "ytd-compact-video-renderer a#thumbnail",
        # Google search results
        "div.g a h3",
        "div[data-sokoban-container] a h3",
        # DuckDuckGo
        "article a[data-testid='result-title-a']",
        # Generic: sizable result links (skip nav/header by text length)
        "a h3",
    ]

    def click_first_result(self, index: int = 1) -> Tuple[bool, str]:
        """Click the Nth search result on the current results page (DOM).
        Skips obviously non-result links (empty/nav) honestly."""
        if not self._try_connect():
            return False, "No DOM control available."
        try:
            for sel in self.RESULT_SELECTORS:
                try:
                    locs = self._page.locator(sel)
                    count = locs.count()
                    if count >= index:
                        target = locs.nth(index - 1)
                        target.click(timeout=3000)
                        title = ""
                        try:
                            title = (target.get_attribute("title") or "").strip()
                        except Exception:
                            pass
                        if not title:
                            try:
                                title = (target.inner_text() or "").strip()[:80]
                            except Exception:
                                pass
                        return True, f"Opened search result #{index}" + (f": {title}." if title else ".")
                except Exception:
                    continue
            # Generic fallback: visible links with meaningful text, in document order
            try:
                links = self._page.locator("a:visible")
                n = links.count()
                meaningful = 0
                for i in range(min(n, 60)):
                    link = links.nth(i)
                    try:
                        text = (link.inner_text() or "").strip()
                    except Exception:
                        continue
                    if len(text) < 15:
                        continue  # nav links, buttons, icons
                    meaningful += 1
                    if meaningful == index:
                        link.click(timeout=3000)
                        return True, f"Opened link result #{index}: {text[:80]}."
            except Exception:
                pass
            return False, f"Only found fewer than {index} result links on the page."
        except Exception as e:
            return False, f"Result click failed: {e}"

    def fill_field(self, placeholder_or_label: str, value: str) -> Tuple[bool, str]:
        """Fill a text input by placeholder/label/aria (DOM)."""
        if not self._try_connect():
            return False, "No DOM control available."
        try:
            for sel_type in ["placeholder", "label"]:
                loc = self._page.get_by_text(placeholder_or_label) if sel_type == "label" else None
                if sel_type == "placeholder":
                    loc = self._page.locator(f"[placeholder*='{placeholder_or_label}' i]").first
                try:
                    if loc and loc.count() > 0:
                        loc.fill(value, timeout=3000)
                        return True, f"Filled '{placeholder_or_label}'."
                except Exception:
                    continue
            # Generic: first empty visible input
            inputs = self._page.locator("input[type=text]:visible, textarea:visible")
            if inputs.count() > 0:
                inputs.first.fill(value, timeout=3000)
                return True, "Filled first visible input."
            return False, "No matching input field found."
        except Exception as e:
            return False, f"Field fill failed: {e}"

    def press_key(self, key: str) -> Tuple[bool, str]:
        if not self._try_connect():
            return False, "No DOM control available."
        try:
            self._page.keyboard.press(key)
            return True, f"Pressed {key} in page."
        except Exception as e:
            return False, f"Key press failed: {e}"

    def download(self, url: str, dest_dir: Optional[str] = None) -> Tuple[bool, str, Optional[str]]:
        """
        Download a file. Native fetch first (fast, no browser), Playwright
        download handling second, curl third.
        """
        dest_dir = dest_dir or str(Path.home() / "Downloads")
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        filename = os.path.basename(urllib.parse.urlparse(url).path) or "download"
        dest_path = str(Path(dest_dir) / filename)

        # 1. Native urllib (fast path for direct URLs)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as out:
                out.write(resp.read())
            if Path(dest_path).stat().st_size > 0:
                return True, f"Downloaded {filename} to {dest_dir}.", dest_path
        except Exception as e:
            logger.debug("Native download failed (%s); trying alternatives", e)

        # 2. Playwright download handling
        if self._try_connect():
            try:
                with self._page.expect_download(timeout=20000) as dl_info:
                    self._page.goto(url, timeout=15000)
                download = dl_info.value
                download.save_as(dest_path)
                return True, f"Downloaded {filename} to {dest_dir}.", dest_path
            except Exception as e:
                logger.debug("Playwright download failed: %s", e)

        # 3. curl
        curl = shutil.which("curl")
        if curl:
            try:
                res = subprocess.run(
                    [curl, "-L", "-o", dest_path, "--max-time", "60", url],
                    capture_output=True, timeout=70,
                )
                if res.returncode == 0 and Path(dest_path).exists() and Path(dest_path).stat().st_size > 0:
                    return True, f"Downloaded {filename} to {dest_dir} (curl).", dest_path
            except Exception as e:
                logger.debug("curl download failed: %s", e)

        return False, f"Failed to download {url}.", None

    def close(self) -> None:
        self._disconnect()
