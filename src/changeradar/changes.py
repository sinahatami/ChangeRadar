"""Parsing helpers for natural-language changes and unified diffs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .models import ParsedFile
from .retrieval import CodeChunk

DIFF_FILE_RE = re.compile(r"^diff --git a/(.*?) b/(.*?)$")
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


@dataclass
class ChangeRequest:
    raw_text: str
    description: str
    changed_files: set[str] = field(default_factory=set)
    changed_lines: dict[str, set[int]] = field(default_factory=dict)
    added_lines: list[str] = field(default_factory=list)
    changed_symbols: set[str] = field(default_factory=set)

    @property
    def retrieval_query(self) -> str:
        if self.description.strip():
            return self.description.strip()
        return "\n".join(self.added_lines).strip() or self.raw_text.strip()


def _safe_diff_path(path: str) -> str | None:
    if path == "/dev/null" or "\\" in path:
        return None
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts:
        return None
    return pure.as_posix()


def parse_change(
    text: str, parsed_files: list[ParsedFile], chunks: list[CodeChunk]
) -> ChangeRequest:
    """Extract changed paths/lines when text is a diff and match known symbols."""
    is_diff = "diff --git " in text or "@@ " in text
    request = ChangeRequest(raw_text=text, description="" if is_diff else text)
    current_path: str | None = None
    new_line: int | None = None
    for line in text.splitlines():
        match = DIFF_FILE_RE.match(line)
        if match:
            current_path = _safe_diff_path(match.group(2))
            new_line = None
            if current_path:
                request.changed_files.add(current_path)
            continue
        if line.startswith("+++ "):
            path = _safe_diff_path(line[4:].removeprefix("b/"))
            if path:
                current_path = path
                request.changed_files.add(path)
            elif line[4:] == "/dev/null":
                current_path = None
            new_line = None
            continue
        hunk = HUNK_RE.match(line)
        if hunk:
            new_line = int(hunk.group(1))
            continue
        if new_line is None or current_path is None or not line:
            continue
        marker = line[0]
        if marker == "+" and not line.startswith("+++"):
            request.changed_lines.setdefault(current_path, set()).add(new_line)
            request.added_lines.append(line[1:])
            new_line += 1
        elif marker == " ":
            new_line += 1
        elif marker == "-" and not line.startswith("---"):
            continue

    symbols_by_path: dict[str, list[tuple[int, int, str]]] = {}
    for chunk in chunks:
        symbols_by_path.setdefault(chunk.path, []).append(
            (chunk.start_line, chunk.end_line, chunk.symbol_name)
        )
    for path, lines in request.changed_lines.items():
        for start, end, symbol in symbols_by_path.get(path, []):
            if any(start <= line <= end for line in lines):
                request.changed_symbols.add(symbol)

    if not is_diff and text:
        lowered = text.lower()
        for chunk in chunks:
            name = chunk.symbol_name.split(".")[-1]
            if re.search(rf"(?<![\w]){re.escape(name.lower())}(?![\w])", lowered):
                request.changed_symbols.add(chunk.symbol_name)
            if chunk.path in text:
                request.changed_files.add(chunk.path)
    return request
