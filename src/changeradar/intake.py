"""Read-only, bounded source intake. Repository code is never executed."""

from __future__ import annotations

import io
import os
import stat
import zipfile
from pathlib import Path, PurePosixPath

from .models import SourceFile

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "ENV",
    "__pycache__",
    "site-packages",
    "node_modules",
    "vendor",
    "build",
    "dist",
    "target",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".nox",
    "coverage",
    "htmlcov",
}
MAX_FILES = 1_000
MAX_FILE_BYTES = 1_000_000
MAX_TOTAL_BYTES = 20_000_000
MAX_ZIP_BYTES = 30_000_000


class IntakeError(ValueError):
    """Raised when input exceeds limits or contains unsafe archive entries."""


def _decode(path: str, data: bytes) -> SourceFile | None:
    if len(data) > MAX_FILE_BYTES:
        return None
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if "\x00" in content:
        return None
    return SourceFile(path=path, content=content)


def scan_directory(root: str | Path) -> list[SourceFile]:
    """Read eligible .py files beneath root, skipping symlinks and ignored trees."""
    base = Path(root).expanduser().resolve(strict=True)
    if not base.is_dir():
        raise IntakeError("Repository path must be a directory")
    results: list[SourceFile] = []
    total = 0
    for current, dirs, names in os.walk(base, followlinks=False):
        current_path = Path(current)
        dirs[:] = sorted(
            name
            for name in dirs
            if name not in IGNORED_DIRS and not (current_path / name).is_symlink()
        )
        for name in sorted(names):
            candidate = current_path / name
            if candidate.suffix != ".py" or candidate.is_symlink():
                continue
            try:
                if not stat.S_ISREG(candidate.lstat().st_mode):
                    continue
                size = candidate.stat().st_size
                if size > MAX_FILE_BYTES:
                    continue
                if len(results) >= MAX_FILES or total + size > MAX_TOTAL_BYTES:
                    raise IntakeError("Repository exceeds source file limits")
                data = candidate.read_bytes()
            except OSError:
                continue
            source = _decode(candidate.relative_to(base).as_posix(), data)
            if source is not None:
                results.append(source)
                total += len(data)
    return results


def scan_zip(data: bytes) -> list[SourceFile]:
    """Read eligible .py entries from an uploaded ZIP without extracting it."""
    if len(data) > MAX_ZIP_BYTES:
        raise IntakeError("ZIP upload exceeds size limit")
    results: list[SourceFile] = []
    total = 0
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError) as exc:
        raise IntakeError("Invalid ZIP archive") from exc
    with archive:
        entries = archive.infolist()
        if len(entries) > MAX_FILES * 5:
            raise IntakeError("ZIP contains too many entries")
        seen: set[str] = set()
        for info in entries:
            name = info.filename
            path = PurePosixPath(name)
            mode = info.external_attr >> 16
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise IntakeError(f"Unsafe ZIP path: {name}")
            if stat.S_ISLNK(mode):
                raise IntakeError(f"ZIP symlinks are not allowed: {name}")
            if info.is_dir() or any(part in IGNORED_DIRS for part in path.parts):
                continue
            if path.suffix != ".py":
                continue
            if name in seen:
                raise IntakeError(f"Duplicate ZIP path: {name}")
            seen.add(name)
            if info.flag_bits & 0x1:
                raise IntakeError("Encrypted ZIP entries are not supported")
            if info.file_size > MAX_FILE_BYTES:
                continue
            if len(results) >= MAX_FILES or total + info.file_size > MAX_TOTAL_BYTES:
                raise IntakeError("ZIP exceeds source file limits")
            try:
                with archive.open(info) as stream:
                    raw = stream.read(MAX_FILE_BYTES + 1)
            except (RuntimeError, zipfile.BadZipFile, OSError) as exc:
                raise IntakeError(f"Could not read ZIP entry: {name}") from exc
            if len(raw) > MAX_FILE_BYTES:
                continue
            if total + len(raw) > MAX_TOTAL_BYTES:
                raise IntakeError("ZIP exceeds source file limits")
            source = _decode(path.as_posix(), raw)
            if source is not None:
                results.append(source)
                total += len(raw)
    return results
