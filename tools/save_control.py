"""
Save/Export Tool for Laya Desktop Agent.

"Save it as AVIF", "save as PNG in my Wallpapers folder".

Strategy (native first, GUI only when necessary):
  1. If the source file is known and Pillow supports the target format,
     convert directly on disk — fast, deterministic, verifiable.
  2. Otherwise drive the already-open application's Save dialog via
     keyboard (Ctrl+Shift+S → filename → Enter) and verify by polling the
     destination for the new file (honest verification, no fake success).
"""

from pathlib import Path
from typing import Callable, Optional
import logging
import re
import time

from tools.base import Tool, ToolResult, SafetyTier
from tools.file_control import get_file_resolver

logger = logging.getLogger(__name__)

# Formats Pillow can write
PIL_FORMATS = {
    "avif": "AVIF", "png": "PNG", "jpg": "JPEG", "jpeg": "JPEG",
    "webp": "WEBP", "bmp": "BMP", "tiff": "TIFF", "gif": "GIF", "ico": "ICO",
    "pdf": "PDF", "tga": "TGA",
}

IMAGE_SOURCE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif", ".avif", ".tga"}


class SaveAsTool(Tool):
    name = "save_as"
    description = "Save/export the current file in a given format and optional location (e.g. 'save it as AVIF')"
    safety_tier = SafetyTier.SAFE

    def __init__(self, type_fn: Optional[Callable[[str], bool]] = None, press_fn: Optional[Callable[[str], bool]] = None):
        """type_fn/press_fn: injected keyboard injection for the GUI fallback path."""
        self._type = type_fn
        self._press = press_fn

    def execute(
        self,
        format: str = "",
        fmt: str = "",
        location: str = "",
        source_path: str = "",
        last_referenced: Optional[str] = None,
        wait_timeout_sec: float = 12.0,
        **kwargs,
    ) -> ToolResult:
        target_format = (format or fmt or "").strip().lower().lstrip(".")
        if not target_format:
            return ToolResult(
                success=False,
                message="No target format specified. What format should I save it as?",
                error="MissingFormat",
                data={"needs_clarification": True},
            )

        resolver = get_file_resolver()
        src = Path(source_path) if source_path else None
        if src is None or not src.exists():
            src = resolver.resolve_target("this file", last_referenced=last_referenced)
        if src is None or not src.exists():
            src = resolver._newest_file("image", None)
        if src is None or not src.exists():
            return ToolResult(
                success=False,
                message="No source file found to save.",
                error="SourceNotFound",
            )

        dst_dir = resolver.resolve_folder(location) if location else src.parent
        if dst_dir is None and location:
            clean = re.sub(r"^(my|the)\s+", "", location.lower().strip())
            clean = re.sub(r"\s+(folder|directory)$", "", clean).strip()
            dst_dir = Path.home() / (clean.capitalize() if clean else "Saved")
            dst_dir.mkdir(parents=True, exist_ok=True)
        dst_dir = dst_dir or src.parent

        new_name = src.stem + "." + target_format
        dst_path = dst_dir / new_name

        # ------------------------------------------------------------------
        # Path 1: Native conversion (Pillow)
        # ------------------------------------------------------------------
        if target_format in PIL_FORMATS and src.suffix.lower() in IMAGE_SOURCE_EXTS:
            try:
                from PIL import Image
                with Image.open(src) as img:
                    # Handle alpha channel requirements
                    if target_format in ("jpg", "jpeg") and img.mode in ("RGBA", "P", "LA"):
                        img = img.convert("RGB")
                    save_kwargs = {}
                    if target_format == "avif":
                        save_kwargs["quality"] = 85
                    img.save(dst_path, PIL_FORMATS[target_format], **save_kwargs)
                if dst_path.exists() and dst_path.stat().st_size > 0:
                    return ToolResult(
                        success=True,
                        message=f"Saved as {target_format.upper()}: {dst_path.name} in {dst_dir.name}.",
                        data={"path": str(dst_path), "format": target_format, "method": "native_pillow"},
                        observation_required=False,
                    )
            except ImportError:
                logger.info("Pillow not installed; using GUI save-dialog path")
            except Exception as e:
                logger.warning("Native conversion failed (%s); trying GUI save dialog", e)

        # ------------------------------------------------------------------
        # Path 2: GUI Save dialog via keyboard injection
        # ------------------------------------------------------------------
        if self._type and self._press:
            try:
                baseline = set(dst_dir.glob(f"*.{target_format}")) if dst_dir.exists() else set()
                # Trigger Save As
                self._press("ctrl+shift+s")
                time.sleep(1.0)
                # Overwrite the suggested filename
                self._type(str(dst_path))
                time.sleep(0.2)
                self._press("Return")

                # Honest verification: poll for the new file
                deadline = time.time() + wait_timeout_sec
                while time.time() < deadline:
                    if dst_path.exists() and dst_path.stat().st_size > 0:
                        return ToolResult(
                            success=True,
                            message=f"Saved as {target_format.upper()} via save dialog: {dst_path.name}.",
                            data={"path": str(dst_path), "format": target_format, "method": "gui_dialog"},
                            observation_required=False,
                        )
                    time.sleep(0.3)
                new_files = set(dst_dir.glob(f"*.{target_format}")) - baseline if dst_dir.exists() else set()
                if new_files:
                    newest = max(new_files, key=lambda p: p.stat().st_mtime)
                    return ToolResult(
                        success=True,
                        message=f"Saved as {target_format.upper()}: {newest.name}.",
                        data={"path": str(newest), "format": target_format, "method": "gui_dialog"},
                        observation_required=False,
                    )
                return ToolResult(
                    success=False,
                    message=f"Save dialog path did not produce a {target_format} file.",
                    error="SaveDialogFailed",
                )
            except Exception as e:
                return ToolResult(success=False, message=f"GUI save failed: {e}", error=str(e))

        return ToolResult(
            success=False,
            message=f"Cannot save as {target_format.upper()}: no converter available (install Pillow) and no keyboard automation.",
            error="NoSaveMechanism",
        )
