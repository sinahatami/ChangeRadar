"""Shared data structures used by scanning and static analysis."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceFile:
    path: str
    content: str


@dataclass(frozen=True)
class Symbol:
    name: str
    qualified_name: str
    kind: str
    path: str
    start_line: int
    end_line: int
    calls: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()


@dataclass
class ParsedFile:
    path: str
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class DependencyEdge:
    source: str
    target: str
    kind: str
    evidence_path: str
    evidence_line: int


@dataclass
class Analysis:
    files: list[ParsedFile]
    dependencies: list[DependencyEdge]
