# ChangeRadar

Local-first code change impact analysis using dependency-aware retrieval and RAG.

## Phase 1: source scanning and static analysis

Phase 1 provides read-only Python source intake, AST symbol extraction, and an
initial dependency graph. It does not execute analyzed repository code. ZIP
uploads are read directly from the archive and are never extracted.

The scanner currently considers UTF-8 `.py` files only. It skips common virtual
environment, dependency, generated, and VCS directories; symlinks; binary or
invalid UTF-8 files; and files larger than 1 MB. Scans are bounded to 1,000
source files and 20 MB of source. ZIP uploads are bounded to 30 MB and reject
path traversal, symlink, encrypted, and duplicate Python entries.

AST analysis reports functions, async functions, classes, line spans, calls,
and imports. The dependency graph links imports that match scanned modules and
calls only when a name resolves to one scanned symbol. Dynamic dispatch,
reflection, conditional imports, and ambiguous names are intentionally left
unresolved, so this graph is a useful lead rather than a complete call graph.

## Local verification

Python 3.10 or newer is required. The Phase 1 application has no runtime
dependencies. Install the lightweight development tools in an isolated
environment with:

```bash
python3 -m pip install -e ".[dev]"
```

Run lint, formatting validation, and tests with:

```bash
ruff check .
ruff format --check .
python3 -m pytest
```

Install the Git hooks once per checkout and run them across the repository with:

```bash
pre-commit install
pre-commit run --all-files
```

The package source is under `src/changeradar`. The next phases can add retrieval,
impact ranking, evaluation, and the Streamlit interface on top of these data
structures without introducing a database or agent framework.
