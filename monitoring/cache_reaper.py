"""
Cache Reaper for Laya Desktop Agent.
Enforces strict disk quotas and TTL eviction for temporary audio and screenshot buffers.
Never touches files outside the agent's dedicated cache directory.
"""

from pathlib import Path
from typing import Optional
import logging
import os
import time
from core.config import AgentConfig

logger = logging.getLogger(__name__)


class CacheReaper:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.cache_dir = config.cache_dir.resolve()
        self.audio_dir = config.audio_cache_dir.resolve()
        self.screenshot_dir = config.screenshot_cache_dir.resolve()

    def reap_all(self) -> int:
        """Run a full TTL and size-based cleanup pass across all cache directories."""
        now = time.time()
        removed_count = 0

        # 1. Clean audio cache by TTL
        removed_count += self._clean_dir_by_age(
            self.audio_dir, max_age_sec=self.config.cache.audio_ttl_sec, now=now
        )

        # 2. Clean screenshot cache by TTL
        removed_count += self._clean_dir_by_age(
            self.screenshot_dir, max_age_sec=self.config.cache.screenshot_ttl_sec, now=now
        )

        # 3. Check total cache size and trim oldest if exceeding max_cache_mb
        removed_count += self._enforce_size_limit()

        return removed_count

    def reap_file(self, file_path: Path) -> bool:
        """Safely delete a specific temporary file immediately post-processing."""
        try:
            p = file_path.resolve()
            # Security assertion: ensure path is strictly inside cache_dir
            if not str(p).startswith(str(self.cache_dir)):
                logger.error("Refusing to delete file outside cache: %s", p)
                return False

            if p.exists() and p.is_file():
                p.unlink()
                logger.debug("Reaped temporary file: %s", p)
                return True
        except Exception as e:
            logger.warning("Failed to reap temporary file %s: %s", file_path, e)
        return False

    def _clean_dir_by_age(self, directory: Path, max_age_sec: float, now: float) -> int:
        if not directory.exists():
            return 0

        removed = 0
        for entry in directory.iterdir():
            if entry.is_file() and entry.name != ".gitkeep":
                try:
                    stat = entry.stat()
                    if (now - stat.st_mtime) > max_age_sec:
                        entry.unlink()
                        removed += 1
                        logger.debug("Evicted aged cache file: %s", entry.name)
                except Exception as e:
                    logger.debug("Could not inspect/remove file %s: %s", entry, e)
        return removed

    def _enforce_size_limit(self) -> int:
        max_bytes = self.config.cache.max_cache_mb * 1024 * 1024
        model_dir = self.config.model_cache_dir.resolve()
        total_bytes = 0
        file_list = []
        for p in self.cache_dir.rglob("*"):
            if p.is_file() and p.name != ".gitkeep":
                # NEVER delete model files or weights!
                try:
                    resolved = p.resolve()
                    if str(resolved).startswith(str(model_dir)):
                        continue
                except Exception:
                    continue

                try:
                    s = p.stat()
                    total_bytes += s.st_size
                    file_list.append((s.st_mtime, p, s.st_size))
                except Exception:
                    pass

        if total_bytes <= max_bytes:
            return 0

        # Sort by modification time ascending (oldest first)
        file_list.sort(key=lambda x: x[0])
        removed = 0
        bytes_to_free = total_bytes - (max_bytes * 0.8)  # Free down to 80% limit

        for _, fpath, fsize in file_list:
            if bytes_to_free <= 0:
                break
            try:
                fpath.unlink()
                bytes_to_free -= fsize
                removed += 1
                logger.info("Evicted cache file due to quota: %s", fpath.name)
            except Exception:
                pass

        return removed
