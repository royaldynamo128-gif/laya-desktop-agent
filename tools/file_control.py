"""
File and Folder Control Tools for Laya Desktop Agent.

Natural-language filesystem operations: open, move, copy, rename, delete,
create folder, compress, extract, find. These use reliable native OS
operations directly (never GUI automation when a native operation works),
with a context resolver for referring expressions like "this file".

Safety model (per requirements):
  - Ordinary operations (open/move/copy/rename/create/compress) are autonomous.
  - Destructive operations (delete) are HIGH_RISK and require confirmation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import fnmatch
import logging
import os
import re
import shutil
import subprocess
import time
import zipfile
import tarfile

from tools.base import Tool, ToolResult, SafetyTier

logger = logging.getLogger(__name__)

# Standard user folders for fuzzy name resolution
SEARCH_FOLDERS = ["Downloads", "Desktop", "Documents", "Pictures", "Music", "Videos", "Projects", "Wallpapers", "Home"]

FILE_TYPE_EXTENSIONS = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".avif", ".gif", ".bmp", ".tiff", ".heic"},
    "picture": {".png", ".jpg", ".jpeg", ".webp", ".avif", ".gif", ".bmp", ".tiff"},
    "photo": {".png", ".jpg", ".jpeg", ".webp", ".avif", ".heic", ".gif"},
    "video": {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv"},
    "audio": {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus"},
    "song": {".mp3", ".wav", ".flac", ".ogg", ".m4a"},
    "music": {".mp3", ".wav", ".flac", ".ogg", ".m4a"},
    "document": {".pdf", ".doc", ".docx", ".txt", ".odt", ".rtf"},
    "archive": {".zip", ".tar", ".gz", ".tgz", ".tar.gz", ".bz2", ".xz", ".7z", ".rar"},
    "zip": {".zip", ".tar.gz", ".tgz"},
}


class FileResolver:
    """Resolves natural-language file references to concrete filesystem paths.

    Home-relative directories are computed lazily per call so tests (and
    multi-user setups) can redirect HOME without stale cached paths.
    """

    def __init__(self, base_dirs: Optional[List[str]] = None):
        self._explicit_base_dirs = base_dirs

    @property
    def base_dirs(self) -> List[str]:
        if self._explicit_base_dirs is not None:
            return self._explicit_base_dirs
        home = str(Path.home())
        return [
            f"{home}/Downloads",
            f"{home}/Desktop",
            f"{home}/Documents",
            f"{home}/Pictures",
            f"{home}/Music",
            f"{home}/Videos",
            f"{home}/Wallpapers",
            str(Path.cwd()),
        ]

    def resolve_folder(self, text: str) -> Optional[Path]:
        """Resolve a folder reference like 'my Wallpapers folder' or 'Downloads'."""
        if not text:
            return None
        t = text.lower().strip()
        t = re.sub(r"^(my|the|in|into|to|at)\s+", "", t)
        t = re.sub(r"\s+(folder|directory|dir)$", "", t).strip()

        # Named standard folders
        candidates = [
            Path.home() / t.capitalize() if t else None,
            Path.home() / t.upper(),
            Path.home() / t,
        ]
        # Special cases
        if t in ("home", "~"):
            return Path.home()
        if t == "downloads":
            return Path.home() / "Downloads"

        for c in candidates:
            if c and c.is_dir():
                return c
        # Any existing dir by that name in home
        p = Path.home() / t
        if p.is_dir():
            return p
        return None

    def resolve_target(
        self,
        text: str,
        last_referenced: Optional[str] = None,
        context_dir: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Resolve a file reference to a path. Order:
        1. Referring expressions -> last referenced file / newest matching file
        2. Absolute / relative path that exists
        3. Filename search in standard folders (exact, then fuzzy)
        """
        if not text:
            return None
        t = text.strip()
        lower = t.lower()

        # 1. Referring expressions
        if lower in ("this file", "that file", "it", "this", "that", "the file", "this image", "that image",
                     "the image", "this download", "the download", "the latest download"):
            if last_referenced and Path(last_referenced).exists():
                return Path(last_referenced)
            return self._newest_file(None, context_dir)

        # "this image" / "the image" → newest image file
        kind = self._file_kind(lower)
        if kind and lower.startswith(("this ", "that ", "the ", "my ")):
            newest = self._newest_file(kind, context_dir)
            if newest:
                return newest

        # 2. Direct path
        expanded = Path(os.path.expanduser(t))
        if expanded.exists():
            return expanded
        if context_dir:
            c = Path(context_dir) / t
            if c.exists():
                return c

        # 3. Search by name in standard folders
        for d in self.base_dirs:
            try:
                p = Path(d) / t
                if p.exists():
                    return p
            except Exception:
                pass

        # 4. Fuzzy: search each folder for files containing the phrase
        phrase = re.sub(r"^(the|a|an|my|this|that)\s+", "", lower)
        phrase = re.sub(r"\.(png|jpg|jpeg|webp|pdf|zip|mp4|mp3|txt|docx?|avif)$", "", phrase)
        for d in self.base_dirs:
            matches = self._search_dir(Path(d), phrase)
            if matches:
                # Most recent match wins
                return max(matches, key=lambda p: p.stat().st_mtime)

        return None

    def _file_kind(self, text: str) -> Optional[str]:
        t = re.sub(r"^(this|that|the|my|a|an)\s+", "", text.strip().lower())
        t = re.sub(r"\s+(file|folder)$", "", t).strip()
        for kind in FILE_TYPE_EXTENSIONS:
            if t == kind or t == kind + "s":
                return kind
        return None

    def _newest_file(self, kind: Optional[str], context_dir: Optional[str]) -> Optional[Path]:
        dirs = self.base_dirs
        if context_dir:
            dirs = [context_dir] + [d for d in self.base_dirs if d != context_dir]
        best: Optional[Path] = None
        best_key = (-1, -1, "")
        for d in dirs:
            try:
                for p in Path(d).glob("*"):
                    if not p.is_file():
                        continue
                    if kind:
                        exts = FILE_TYPE_EXTENSIONS.get(kind, set())
                        if p.suffix.lower() not in exts:
                            continue
                    st = p.stat()
                    # Deterministic ordering: mtime, then ctime, then path —
                    # never depends on filesystem iteration order when two
                    # files share a timestamp (coarse-grained filesystems).
                    key = (st.st_mtime_ns, st.st_ctime_ns, str(p))
                    if key > best_key:
                        best_key = key
                        best = p
            except Exception:
                continue
        return best

    def _search_dir(self, directory: Path, phrase: str, limit: int = 200) -> List[Path]:
        if not phrase or not directory.is_dir():
            return []
        matches = []
        try:
            entries = list(directory.iterdir())
        except Exception:
            return []
        for p in entries[:limit]:
            if p.is_file() and phrase in p.stem.lower():
                matches.append(p)
        return matches


_resolver: Optional[FileResolver] = None


def get_file_resolver() -> FileResolver:
    global _resolver
    if _resolver is None:
        _resolver = FileResolver()
    return _resolver


# =============================================================================
# Tools
# =============================================================================


class FileOpenTool(Tool):
    name = "file_open"
    description = "Open a file (image, document, video, etc.) with its default application"
    safety_tier = SafetyTier.SAFE

    def execute(self, target: str = "", path: str = "", last_referenced: Optional[str] = None, **kwargs) -> ToolResult:
        resolver = get_file_resolver()
        resolved = Path(path) if path else resolver.resolve_target(target, last_referenced=last_referenced)
        if not resolved or not resolved.exists():
            return ToolResult(
                success=False,
                message=f"Could not find file matching '{target or path}'.",
                error="FileNotFound",
            )

        opener = shutil.which("xdg-open")
        if not opener:
            return ToolResult(success=False, message="No file opener available (xdg-open).", error="NoOpener")

        try:
            subprocess.Popen(
                [opener, str(resolved)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return ToolResult(
                success=True,
                message=f"Opened file '{resolved.name}'.",
                data={"path": str(resolved), "name": resolved.name},
                observation_required=False,  # native operation self-verified (file existed & opener ran)
            )
        except Exception as e:
            return ToolResult(success=False, message=f"Failed to open file: {e}", error=str(e))


class FileMoveTool(Tool):
    name = "file_move"
    description = "Move a file or folder to a destination folder (e.g. 'move this file to Wallpapers')"
    safety_tier = SafetyTier.SAFE

    def execute(
        self,
        source: str = "",
        destination: str = "",
        source_path: str = "",
        dest_path: str = "",
        last_referenced: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        resolver = get_file_resolver()
        src = Path(source_path) if source_path else resolver.resolve_target(source, last_referenced=last_referenced)
        dst_dir = Path(dest_path) if dest_path else resolver.resolve_folder(destination)

        if not src or not src.exists():
            return ToolResult(success=False, message=f"Could not find source file '{source or source_path}'.", error="SourceNotFound")
        if not dst_dir:
            # Destination folder does not exist: create it under home (natural intent)
            clean = re.sub(r"^(my|the)\s+", "", destination.lower().strip())
            clean = re.sub(r"\s+(folder|directory)$", "", clean).strip()
            if clean:
                dst_dir = Path.home() / clean.capitalize()
                try:
                    dst_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    return ToolResult(success=False, message=f"Could not create destination folder: {e}", error=str(e))
            else:
                return ToolResult(success=False, message=f"Could not resolve destination folder '{destination}'.", error="DestNotFound")

        try:
            final = dst_dir / src.name
            shutil.move(str(src), str(final))
            return ToolResult(
                success=True,
                message=f"Moved '{src.name}' to {dst_dir.name}.",
                data={"source": str(src), "destination": str(final), "moved_path": str(final)},
                observation_required=False,
            )
        except Exception as e:
            return ToolResult(success=False, message=f"Failed to move file: {e}", error=str(e))


class FileRenameTool(Tool):
    name = "file_rename"
    description = "Rename a file (e.g. 'rename this file to notes.txt')"
    safety_tier = SafetyTier.SAFE

    def execute(
        self,
        source: str = "",
        new_name: str = "",
        source_path: str = "",
        last_referenced: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        if not new_name or not new_name.strip():
            return ToolResult(
                success=False,
                message="No new name specified. What should the file be called?",
                error="MissingNewName",
                data={"needs_clarification": True},
            )

        resolver = get_file_resolver()
        src = Path(source_path) if source_path else resolver.resolve_target(source, last_referenced=last_referenced)
        if not src or not src.exists():
            return ToolResult(success=False, message=f"Could not find file '{source or source_path}'.", error="SourceNotFound")

        new_name = new_name.strip()
        # Preserve extension if the new name lacks one
        if not Path(new_name).suffix and src.suffix:
            new_name = new_name + src.suffix
        target = src.parent / new_name
        if target.exists() and target != src:
            return ToolResult(success=False, message=f"A file named '{new_name}' already exists.", error="NameConflict")

        try:
            src.rename(target)
            return ToolResult(
                success=True,
                message=f"Renamed to '{new_name}'.",
                data={"source": str(src), "destination": str(target), "moved_path": str(target)},
                observation_required=False,
            )
        except Exception as e:
            return ToolResult(success=False, message=f"Failed to rename: {e}", error=str(e))


class FileCopyTool(Tool):
    name = "file_copy"
    description = "Copy a file to a destination folder"
    safety_tier = SafetyTier.SAFE

    def execute(
        self,
        source: str = "",
        destination: str = "",
        source_path: str = "",
        dest_path: str = "",
        last_referenced: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        resolver = get_file_resolver()
        src = Path(source_path) if source_path else resolver.resolve_target(source, last_referenced=last_referenced)
        dst_dir = Path(dest_path) if dest_path else resolver.resolve_folder(destination)
        if not src or not src.exists():
            return ToolResult(success=False, message=f"Could not find file '{source}'.", error="SourceNotFound")
        if not dst_dir or not dst_dir.is_dir():
            return ToolResult(success=False, message=f"Could not resolve destination '{destination}'.", error="DestNotFound")
        try:
            final = dst_dir / src.name
            shutil.copy2(str(src), str(final))
            return ToolResult(
                success=True,
                message=f"Copied '{src.name}' to {dst_dir.name}.",
                data={"source": str(src), "destination": str(final)},
                observation_required=False,
            )
        except Exception as e:
            return ToolResult(success=False, message=f"Failed to copy: {e}", error=str(e))


class FileDeleteTool(Tool):
    name = "file_delete"
    description = "Delete files or folders (HIGH RISK - requires confirmation)"
    safety_tier = SafetyTier.HIGH_RISK

    def execute(self, target: str = "", path: str = "", last_referenced: Optional[str] = None, **kwargs) -> ToolResult:
        resolver = get_file_resolver()
        # Support plural referring targets ("these files", "the temporary files")
        paths = self._resolve_targets(target, path, last_referenced)
        if not paths:
            return ToolResult(success=False, message=f"Could not find files matching '{target or path}'.", error="NotFound")

        deleted = []
        for p in paths:
            try:
                p.unlink()
                deleted.append(str(p))
            except Exception as e:
                return ToolResult(success=False, message=f"Failed to delete {p}: {e}", error=str(e))

        return ToolResult(
            success=True,
            message=f"Deleted {len(deleted)} file(s).",
            data={"deleted": deleted},
            observation_required=False,
        )

    def _resolve_targets(self, target: str, path: str, last_referenced: Optional[str]) -> List[Path]:
        if path:
            p = Path(path)
            return [p] if p.exists() else []
        resolver = get_file_resolver()
        lower = (target or "").lower()

        # "these files" / "them" → all files in context or last referenced
        if lower in ("these files", "them", "those files", "the files"):
            if last_referenced and Path(last_referenced).is_dir():
                return [p for p in Path(last_referenced).iterdir() if p.is_file()]
            if last_referenced and Path(last_referenced).exists():
                return [Path(last_referenced)]
            return []

        # "temporary files" pattern → temp patterns in standard dirs
        temp_m = re.match(r"^(?:these|those|the|my)?\s*(?:temporary|temp)\s+files$", lower.strip())
        if temp_m:
            results = []
            for d in resolver.base_dirs:
                try:
                    for p in Path(d).glob("*"):
                        if p.is_file() and re.search(r"(temp|tmp|~$|^\.#|backup|bak)", p.name.lower()):
                            results.append(p)
                except Exception:
                    pass
            return results

        single = resolver.resolve_target(target, last_referenced=last_referenced)
        return [single] if single else []


class FolderCreateTool(Tool):
    name = "file_create_folder"
    description = "Create a new folder (e.g. 'create a folder called Projects')"
    safety_tier = SafetyTier.SAFE

    def __init__(self, base_dir: Optional[Path] = None):
        self._base_dir = base_dir  # None → resolve HOME lazily at execution time

    def execute(self, name: str = "", path: str = "", **kwargs) -> ToolResult:
        if path:
            target = Path(os.path.expanduser(path))
        elif name:
            base = self._base_dir if self._base_dir is not None else Path.home()
            clean = name.strip()
            target = Path(base) / clean
        else:
            return ToolResult(success=False, message="Folder name required.", error="MissingName")

        if target.exists():
            return ToolResult(
                success=True,
                message=f"Folder '{target.name}' already exists.",
                data={"path": str(target), "already_exists": True},
                observation_required=False,
            )
        try:
            target.mkdir(parents=True, exist_ok=True)
            return ToolResult(
                success=True,
                message=f"Created folder '{target.name}'.",
                data={"path": str(target)},
                observation_required=False,
            )
        except Exception as e:
            return ToolResult(success=False, message=f"Failed to create folder: {e}", error=str(e))


class FileCompressTool(Tool):
    name = "file_compress"
    description = "Compress files or a folder into an archive (zip/tar.gz)"
    safety_tier = SafetyTier.SAFE

    def execute(self, targets: str = "", path: str = "", archive_name: str = "", last_referenced: Optional[str] = None, **kwargs) -> ToolResult:
        resolver = get_file_resolver()
        # Resolve the target set
        lower = (targets or path or "").lower()
        file_list: List[Path] = []
        if lower in ("these files", "them", "those files", "the files") and last_referenced:
            ref = Path(last_referenced)
            if ref.is_dir():
                file_list = [p for p in ref.iterdir() if p.is_file()]
            elif ref.exists():
                file_list = [ref]
        else:
            # Single path? folder?
            resolved = resolver.resolve_target(targets or path, last_referenced=last_referenced)
            if resolved:
                if resolved.is_dir():
                    file_list = [p for p in resolved.iterdir() if p.is_file()]
                    if not file_list:
                        file_list = [resolved]
                else:
                    file_list = [resolved]
            else:
                folder = resolver.resolve_folder(targets or path)
                if folder and folder.is_dir():
                    file_list = [p for p in folder.iterdir() if p.is_file()]

        if not file_list:
            return ToolResult(success=False, message=f"Could not find files matching '{targets or path}'.", error="NotFound")

        # Archive name/location
        if archive_name:
            out_path = Path(os.path.expanduser(archive_name))
            if not out_path.suffix:
                out_path = out_path.with_suffix(".zip")
        else:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = file_list[0].parent / f"archive_{stamp}.zip"

        try:
            with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in file_list:
                    zf.write(f, arcname=f.name)
            return ToolResult(
                success=True,
                message=f"Compressed {len(file_list)} file(s) into '{out_path.name}'.",
                data={"archive": str(out_path), "count": len(file_list)},
                observation_required=False,
            )
        except Exception as e:
            return ToolResult(success=False, message=f"Failed to compress: {e}", error=str(e))


class FileExtractTool(Tool):
    name = "file_extract"
    description = "Extract an archive (zip/tar.gz) into its folder"
    safety_tier = SafetyTier.SAFE

    def execute(self, target: str = "", path: str = "", last_referenced: Optional[str] = None, **kwargs) -> ToolResult:
        resolver = get_file_resolver()
        src = Path(path) if path else resolver.resolve_target(target, last_referenced=last_referenced)
        if not src or not src.exists():
            return ToolResult(success=False, message=f"Could not find archive '{target or path}'.", error="NotFound")

        out_dir = src.parent / src.stem
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            if src.suffix == ".zip":
                with zipfile.ZipFile(src) as zf:
                    zf.extractall(out_dir)
            else:
                with tarfile.open(src) as tf:
                    tf.extractall(out_dir)
            return ToolResult(
                success=True,
                message=f"Extracted '{src.name}' to '{out_dir.name}'.",
                data={"archive": str(src), "output": str(out_dir)},
                observation_required=False,
            )
        except Exception as e:
            return ToolResult(success=False, message=f"Failed to extract: {e}", error=str(e))


class FileFindTool(Tool):
    name = "file_find"
    description = "Find files by criterion (largest, newest, pattern) in standard folders"
    safety_tier = SafetyTier.SAFE

    def execute(self, criterion: str = "largest", pattern: str = "", folder: str = "", top: int = 10, **kwargs) -> ToolResult:
        resolver = get_file_resolver()
        search_dirs = [resolver.resolve_folder(folder)] if folder and resolver.resolve_folder(folder) else [Path(d) for d in resolver.base_dirs]

        results: List[Tuple[Path, float]] = []
        for d in search_dirs:
            try:
                for p in d.glob("*"):
                    if not p.is_file():
                        continue
                    if pattern and not fnmatch.fnmatch(p.name.lower(), f"*{pattern.lower()}*"):
                        continue
                    try:
                        if criterion in ("newest", "recent"):
                            results.append((p, p.stat().st_mtime))
                        else:
                            results.append((p, p.stat().st_size))
                    except Exception:
                        continue
            except Exception:
                continue

        if not results:
            return ToolResult(
                success=True,
                message="No matching files found.",
                data={"files": [], "criterion": criterion},
                observation_required=False,
            )

        results.sort(key=lambda x: x[1], reverse=True)
        top_files = results[: top]
        lines = []
        for p, metric in top_files:
            size = f"{metric / (1024 * 1024):.1f} MB" if criterion not in ("newest", "recent") and metric > 1024 * 1024 else (f"{metric / 1024:.1f} KB" if criterion not in ("newest", "recent") else time.strftime("%H:%M:%S %d %b", time.localtime(metric)))
            lines.append(f"- {p.name} ({size}) in {p.parent.name}")

        largest_path = top_files[0][0]
        return ToolResult(
            success=True,
            message="Found files:\n" + "\n".join(lines),
            data={
                "files": [str(p) for p, _ in top_files],
                "criterion": criterion,
                "top_file": str(largest_path),
            },
            observation_required=False,
        )
