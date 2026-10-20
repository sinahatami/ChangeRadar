"""Validated in-memory analysis tools; tool inputs never access arbitrary paths."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Callable

from .graph import module_name
from .models import Analysis, ParsedFile, SourceFile
from .retrieval import CodeChunk, CodeRetriever


class ToolArgumentError(ValueError):
    """Tool arguments failed validation."""


class ToolRegistry:
    def __init__(
        self,
        sources: list[SourceFile],
        parsed_files: list[ParsedFile],
        chunks: list[CodeChunk],
        analysis: Analysis,
        retriever: CodeRetriever,
        max_calls: int = 12,
        retrieval_method: str = "hybrid",
    ) -> None:
        if retrieval_method not in {"lexical", "semantic", "hybrid"}:
            raise ValueError("retrieval_method must be lexical, semantic, or hybrid")
        self.sources = {source.path: source for source in sources}
        self.parsed_files = parsed_files
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.analysis = analysis
        self.retriever = retriever
        self.retrieval_method = retrieval_method
        self.max_calls = max_calls
        self.calls = 0
        self._tools: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "search_code": self._search_code,
            "inspect_dependencies": self._inspect_dependencies,
            "get_source_snippet": self._get_source_snippet,
            "rank_impact_candidates": self._rank_impact_candidates,
        }

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.calls >= self.max_calls:
            return {"error": "Tool call limit reached"}
        if not isinstance(arguments, dict):
            return {"error": "Tool arguments must be an object"}
        tool = self._tools.get(name)
        if tool is None:
            return {"error": "Unknown tool"}
        self.calls += 1
        try:
            return tool(arguments)
        except ToolArgumentError as exc:
            return {"error": str(exc)}

    def definitions(self) -> list[dict[str, Any]]:
        """OpenAI-compatible function tool definitions."""
        return [
            _tool_schema(
                "search_code",
                "Search analyzed source code using the configured local retrieval method.",
                {
                    "query": _string_schema(),
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                ["query"],
            ),
            _tool_schema(
                "inspect_dependencies",
                "Inspect dependency edges for a known symbol or module.",
                {"symbol": _string_schema()},
                ["symbol"],
            ),
            _tool_schema(
                "get_source_snippet",
                "Retrieve a bounded source line range from an already analyzed file.",
                {
                    "path": _string_schema(),
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                ["path", "start_line", "end_line"],
            ),
            _tool_schema(
                "rank_impact_candidates",
                "Rank potential impacts for a change query and known changed symbols.",
                {
                    "query": _string_schema(),
                    "changed_symbols": {"type": "array", "items": _string_schema(), "maxItems": 20},
                },
                ["query"],
            ),
        ]

    def _search_code(self, args: dict[str, Any]) -> dict[str, Any]:
        _check_keys(args, {"query", "top_k"}, {"query"})
        query = _checked_string(args["query"], "query", 1, 1_000)
        top_k = args.get("top_k", 5)
        if type(top_k) is not int or not 1 <= top_k <= 10:
            raise ToolArgumentError("top_k must be an integer from 1 to 10")
        response = self.retriever.retrieve(query, method=self.retrieval_method, top_k=top_k)
        return {
            "effective_method": response.effective_method,
            "warning": response.warning,
            "results": [_hit_dict(hit) for hit in response.hits],
        }

    def _inspect_dependencies(self, args: dict[str, Any]) -> dict[str, Any]:
        _check_keys(args, {"symbol"}, {"symbol"})
        symbol = _checked_string(args["symbol"], "symbol", 1, 300)
        known = self._known_symbols()
        if symbol not in known:
            raise ToolArgumentError("symbol is not present in the analyzed repository")
        edges = [
            edge
            for edge in self.analysis.dependencies
            if edge.source == symbol
            or edge.target == symbol
            or edge.source.endswith(f":{symbol}")
            or edge.target.endswith(f":{symbol}")
        ]
        return {
            "symbol": symbol,
            "dependencies": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "kind": edge.kind,
                    "evidence_path": edge.evidence_path,
                    "evidence_line": edge.evidence_line,
                }
                for edge in edges
            ],
        }

    def _get_source_snippet(self, args: dict[str, Any]) -> dict[str, Any]:
        _check_keys(args, {"path", "start_line", "end_line"}, {"path", "start_line", "end_line"})
        path = _checked_string(args["path"], "path", 1, 500)
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or "\\" in path:
            raise ToolArgumentError("path must be a repository-relative analyzed path")
        normalized = pure.as_posix()
        source = self.sources.get(normalized)
        if source is None:
            raise ToolArgumentError("path is not an analyzed source file")
        start = args["start_line"]
        end = args["end_line"]
        if type(start) is not int or type(end) is not int or start < 1 or end < start:
            raise ToolArgumentError("line range must use positive integers with end >= start")
        if end - start + 1 > 100:
            raise ToolArgumentError("snippet line range cannot exceed 100 lines")
        lines = source.content.splitlines(keepends=True)
        if end > len(lines):
            raise ToolArgumentError("line range exceeds the source file")
        return {
            "path": normalized,
            "start_line": start,
            "end_line": end,
            "text": "".join(lines[start - 1 : end]),
        }

    def _rank_impact_candidates(self, args: dict[str, Any]) -> dict[str, Any]:
        _check_keys(args, {"query", "changed_symbols"}, {"query"})
        query = _checked_string(args["query"], "query", 1, 1_000)
        changed = args.get("changed_symbols", [])
        if not isinstance(changed, list) or len(changed) > 20:
            raise ToolArgumentError("changed_symbols must be a list of at most 20 symbols")
        known = self._known_symbols()
        if any(not isinstance(item, str) or item not in known for item in changed):
            raise ToolArgumentError("changed_symbols must contain only known symbols")
        response = self.retriever.retrieve(
            query, method=self.retrieval_method, top_k=10, changed_symbols=set(changed)
        )
        return {
            "effective_method": response.effective_method,
            "warning": response.warning,
            "candidates": [_hit_dict(hit) for hit in response.hits],
        }

    def _known_symbols(self) -> set[str]:
        known = set(self.chunks)
        for chunk in self.chunks.values():
            known.add(chunk.symbol_name)
            known.add(f"{_module_for_path(chunk.path)}:{chunk.symbol_name}")
        for parsed in self.parsed_files:
            module = _module_for_path(parsed.path)
            known.add(module)
            known.update(f"{module}:{symbol.qualified_name}" for symbol in parsed.symbols)
        return known


def _module_for_path(path: str) -> str:
    return module_name(path)


def _hit_dict(hit) -> dict[str, Any]:
    chunk = hit.chunk
    return {
        "chunk_id": chunk.chunk_id,
        "path": chunk.path,
        "symbol": chunk.symbol_name,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "score": hit.score,
        "lexical_score": hit.lexical_score,
        "semantic_score": hit.semantic_score,
        "dependency_boost": hit.dependency_boost,
        "text": chunk.text,
    }


def _check_keys(args: dict[str, Any], allowed: set[str], required: set[str]) -> None:
    if set(args) - allowed:
        raise ToolArgumentError("unexpected tool argument")
    if required - set(args):
        raise ToolArgumentError("missing required tool argument")


def _checked_string(value: Any, name: str, minimum: int, maximum: int) -> str:
    if not isinstance(value, str) or not minimum <= len(value.strip()) <= maximum:
        raise ToolArgumentError(f"{name} must be a string of {minimum} to {maximum} characters")
    return value.strip()


def _string_schema() -> dict[str, str]:
    return {"type": "string"}


def _tool_schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }
