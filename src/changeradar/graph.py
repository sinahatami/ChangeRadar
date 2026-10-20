"""Conservative import and statically named call relationships."""

from pathlib import PurePosixPath

from .models import Analysis, DependencyEdge, ParsedFile


def module_name(path: str) -> str:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def build_analysis(files: list[ParsedFile]) -> Analysis:
    modules = {module_name(file.path): file.path for file in files}
    symbols: dict[str, list[str]] = {}
    for file in files:
        module = module_name(file.path)
        for symbol in file.symbols:
            identity = f"{module}:{symbol.qualified_name}"
            symbols.setdefault(symbol.name, []).append(identity)
            symbols.setdefault(symbol.qualified_name, []).append(identity)
    edges: list[DependencyEdge] = []
    for file in files:
        module = module_name(file.path)
        for imported, line in file.import_sites:
            target_path = modules.get(imported) or modules.get(imported.split(".")[0])
            if target_path and target_path != file.path:
                edges.append(
                    DependencyEdge(module, module_name(target_path), "import", file.path, line)
                )
        for symbol in file.symbols:
            for call, line in symbol.call_sites:
                candidates = symbols.get(call, []) + symbols.get(call.rsplit(".", 1)[-1], [])
                targets = sorted(set(candidates))
                source = f"{module}:{symbol.qualified_name}"
                if len(targets) == 1 and targets[0] != source:
                    edges.append(DependencyEdge(source, targets[0], "call", file.path, line))
    unique = {
        (edge.source, edge.target, edge.kind, edge.evidence_path, edge.evidence_line): edge
        for edge in edges
    }
    return Analysis(
        files=files,
        dependencies=sorted(
            unique.values(), key=lambda edge: (edge.evidence_path, edge.evidence_line, edge.kind)
        ),
    )
