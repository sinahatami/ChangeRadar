#!/usr/bin/env python3
"""Evaluate retrieval methods against the labeled local fixture set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from changeradar.graph import build_analysis
from changeradar.intake import scan_directory
from changeradar.parser import parse_source
from changeradar.retrieval import CodeRetriever, build_code_chunks

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_REPO = ROOT / "tests" / "fixtures" / "retrieval_repo"
LABELS = ROOT / "tests" / "fixtures" / "retrieval_labels.json"


def metrics(retriever: CodeRetriever, labels: list[dict[str, object]], method: str) -> dict:
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    effective_methods: set[str] = set()
    warning: str | None = None
    for case in labels:
        relevant = set(case["relevant"])
        response = retriever.retrieve(str(case["query"]), method=method, top_k=5)
        effective_methods.add(response.effective_method)
        warning = warning or response.warning
        ranked = [hit.chunk.chunk_id for hit in response.hits]
        recalls.append(len(relevant.intersection(ranked)) / len(relevant) if relevant else 0.0)
        first_rank = next(
            (index for index, item in enumerate(ranked, start=1) if item in relevant), None
        )
        reciprocal_ranks.append(1 / first_rank if first_rank else 0.0)
    result = {
        "requested_method": method,
        "effective_method": ",".join(sorted(effective_methods)),
        "queries": len(labels),
        "Recall@5": sum(recalls) / len(recalls) if recalls else 0.0,
        "MRR": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
    }
    if warning:
        result["note"] = warning
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Enable the local Sentence Transformers model (may download weights if uncached)",
    )
    args = parser.parse_args()
    sources = scan_directory(FIXTURE_REPO)
    parsed = [parse_source(source) for source in sources]
    analysis = build_analysis(parsed)
    chunks = build_code_chunks(sources, parsed)
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    retriever = CodeRetriever(chunks, analysis=analysis)
    if args.semantic:
        from changeradar.retrieval import SentenceTransformerEncoder

        retriever.encoder = SentenceTransformerEncoder()
    print("Fixture-based retrieval results (computed from current rankings):")
    for method in ("lexical", "semantic", "hybrid"):
        print(json.dumps(metrics(retriever, labels, method), sort_keys=True))


if __name__ == "__main__":
    main()
