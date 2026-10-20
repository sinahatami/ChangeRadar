#!/usr/bin/env python3
"""Exercise the real local Sentence Transformers path on the bundled fixture."""

from __future__ import annotations

import json
from pathlib import Path

from changeradar.graph import build_analysis
from changeradar.intake import scan_directory
from changeradar.parser import parse_source
from changeradar.retrieval import CodeRetriever, SentenceTransformerEncoder, build_code_chunks

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_REPO = ROOT / "tests" / "fixtures" / "retrieval_repo"


def main() -> None:
    sources = scan_directory(FIXTURE_REPO)
    parsed = [parse_source(source) for source in sources]
    encoder = SentenceTransformerEncoder()
    retriever = CodeRetriever(
        build_code_chunks(sources, parsed),
        analysis=build_analysis(parsed),
        encoder=encoder,
    )
    response = retriever.retrieve(
        "change how calculate_tax handles a zero tax rate", method="semantic", top_k=5
    )

    if response.effective_method != "semantic" or encoder._model is None:
        detail = response.warning or "The semantic model did not load."
        raise SystemExit(
            f"Semantic smoke test failed: {detail}\n"
            'Install with: python -m pip install -e ".[semantic]"; '
            "then rerun (the first run may download model weights)."
        )

    source_by_path = {source.path: source for source in sources}
    for hit in response.hits:
        source = source_by_path.get(hit.chunk.path)
        if source is None:
            raise SystemExit(f"Invalid result path: {hit.chunk.path}")
        line_count = len(source.content.splitlines())
        if not (1 <= hit.chunk.start_line <= hit.chunk.end_line <= line_count):
            raise SystemExit(
                f"Invalid result line range: {hit.chunk.path}:"
                f"{hit.chunk.start_line}-{hit.chunk.end_line}"
            )

    print(
        json.dumps(
            {
                "status": "passed",
                "model": encoder.model_name,
                "device": "cpu",
                "effective_method": response.effective_method,
                "results": [
                    {
                        "path": hit.chunk.path,
                        "symbol": hit.chunk.symbol_name,
                        "lines": [hit.chunk.start_line, hit.chunk.end_line],
                        "score": hit.score,
                        "semantic_score": hit.semantic_score,
                    }
                    for hit in response.hits
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
