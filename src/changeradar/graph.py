"""Conservative import and statically named call relationships."""

from pathlib import PurePosixPath

from .models import Analysis, DependencyEdge, ParsedFile


def _module(path: str) -> str:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def build_analysis(files: list[ParsedFile]) -> Analysis:
    modules = {_module(file.path): file.path for file in files}
    symbols: dict[str, list[str]] = {}
    for file in files:
        module = _module(file.path)
        for symbol in file.symbols:
            identity = f"{module}:{symbol.qualified_name}"
            symbols.setdefault(symbol.name, []).append(identity)
            symbols.setdefault(symbol.qualified_name, []).append(identity)

    edges: list[DependencyEdge] = []
    for file in files:
        module = _module(file.path)
        for imported in file.imports:
            target_path = modules.get(imported)
            if target_path is None:
                target_path = modules.get(imported.split(".")[0])
            if target_path and target_path != file.path:
                edges.append(DependencyEdge(module, _module(target_path), "import", file.path, 1))
        for symbol in file.symbols:
            for call in symbol.calls:
                candidates = symbols.get(call, [])
                if call.rsplit(".", 1)[-1] in symbols:
                    candidates = candidates + symbols[call.rsplit(".", 1)[-1]]
                targets = sorted(set(candidates))
                source = f"{module}:{symbol.qualified_name}"
                if len(targets) == 1 and targets[0] != source:
                    edges.append(
                        DependencyEdge(source, targets[0], "call", file.path, symbol.start_line)
                    )
    unique = {
        (edge.source, edge.target, edge.kind, edge.evidence_path, edge.evidence_line): edge
        for edge in edges
    }
    return Analysis(
        files=files,
        dependencies=sorted(
            unique.values(), key=lambda e: (e.evidence_path, e.evidence_line, e.kind, e.target)
        ),
    )
