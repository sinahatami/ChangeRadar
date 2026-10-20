"""Best-effort Python AST extraction for symbols, calls, and imports."""

import ast

from .models import ParsedFile, SourceFile, Symbol


class _References(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: set[str] = set()
        self.call_sites: set[tuple[str, int]] = set()
        self.imports: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        name = _dotted_name(node.func)
        if name:
            self.calls.add(name)
            self.call_sites.add((name, node.lineno))
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.update(alias.name for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.add(node.module)


def _dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def parse_source(source: SourceFile) -> ParsedFile:
    result = ParsedFile(path=source.path)
    try:
        tree = ast.parse(source.content, filename=source.path)
    except (SyntaxError, ValueError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        return result
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.imports.extend(alias.name for alias in node.names)
            result.import_sites.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.imports.append(node.module)
            result.import_sites.append((node.module, node.lineno))

    def collect(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualified = f"{prefix}.{node.name}" if prefix else node.name
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                visitor = _References()
                for child in node.body:
                    visitor.visit(child)
                result.symbols.append(
                    Symbol(
                        name=node.name,
                        qualified_name=qualified,
                        kind=kind,
                        path=source.path,
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        calls=tuple(sorted(visitor.calls)),
                        imports=tuple(sorted(visitor.imports)),
                        call_sites=tuple(sorted(visitor.call_sites)),
                    )
                )
                if isinstance(node, ast.ClassDef):
                    collect(node.body, qualified)

    collect(tree.body)
    result.imports = sorted(set(result.imports))
    result.import_sites = sorted(set(result.import_sites))
    return result
