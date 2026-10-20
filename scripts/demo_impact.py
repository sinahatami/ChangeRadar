#!/usr/bin/env python3
"""Run a local impact analysis against a repository without executing its code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from changeradar.graph import build_analysis
from changeradar.intake import scan_directory
from changeradar.llm import OpenAICompatibleProvider
from changeradar.parser import parse_source
from changeradar.rag import ImpactWorkflow
from changeradar.retrieval import CodeRetriever, build_code_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path, help="Path to an analyzed Python repository")
    parser.add_argument("change", help="Natural-language change description or unified diff")
    parser.add_argument(
        "--allow-external",
        action="store_true",
        help="Allow the explicitly configured LLM provider to receive redacted evidence",
    )
    args = parser.parse_args()

    sources = scan_directory(args.repository)
    parsed = [parse_source(source) for source in sources]
    analysis = build_analysis(parsed)
    chunks = build_code_chunks(sources, parsed)
    retriever = CodeRetriever(chunks, analysis=analysis)
    provider = OpenAICompatibleProvider.from_env() if args.allow_external else None
    workflow = ImpactWorkflow(
        sources,
        parsed,
        analysis,
        chunks,
        retriever,
        provider=provider,
        allow_external=provider is not None and args.allow_external,
    )
    print(json.dumps(workflow.analyze(args.change).to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
