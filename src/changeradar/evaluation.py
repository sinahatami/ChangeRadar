"""Offline evaluation for direct, dependency-linked, and context-only impact reports."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .graph import build_analysis
from .intake import scan_directory
from .parser import parse_source
from .rag import ImpactWorkflow
from .retrieval import CodeRetriever, Encoder, SentenceTransformerEncoder, build_code_chunks

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_REPOSITORY = ROOT / "tests" / "fixtures" / "retrieval_repo"
DEFAULT_LABELS = ROOT / "tests" / "fixtures" / "impact_eval_labels.json"


def load_impact_labels(path: Path = DEFAULT_LABELS) -> list[dict[str, Any]]:
    """Load and minimally validate the labeled impact evaluation cases."""
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("impact evaluation labels must be a non-empty JSON list")
    seen: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"impact evaluation case {index} must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError(f"impact evaluation case {index} has a missing or duplicate id")
        seen.add(case_id)
        if not isinstance(case.get("change"), str) or not case["change"].strip():
            raise ValueError(f"impact evaluation case {case_id} has no change description")
        for field in (
            "expected_direct",
            "expected_dependency_linked",
            "expected_context_only",
            "expected_test_markers",
        ):
            values = case.get(field)
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise ValueError(f"impact evaluation case {case_id} has invalid {field}")
        if type(case.get("expected_ambiguous")) is not bool:
            raise ValueError(f"impact evaluation case {case_id} has invalid expected_ambiguous")
    return cases


def run_impact_evaluation(
    method: str = "lexical",
    cases: list[dict[str, Any]] | None = None,
    encoder: Encoder | None = None,
) -> dict[str, Any]:
    """Run actual reports and calculate labeled metrics; lexical mode needs no model."""
    if method not in {"lexical", "semantic", "hybrid"}:
        raise ValueError("method must be lexical, semantic, or hybrid")
    labels = cases if cases is not None else load_impact_labels()
    sources = scan_directory(SAMPLE_REPOSITORY)
    parsed_files = [parse_source(source) for source in sources]
    analysis = build_analysis(parsed_files)
    chunks = build_code_chunks(sources, parsed_files)
    if method != "lexical" and encoder is None:
        encoder = SentenceTransformerEncoder()
    retriever = CodeRetriever(chunks, analysis=analysis, encoder=encoder)
    workflow = ImpactWorkflow(
        sources,
        parsed_files,
        analysis,
        chunks,
        retriever,
        retrieval_method=method,
    )
    source_by_path = {source.path: source for source in sources}
    evidence_counts: Counter[str] = Counter()
    candidate_evidence_valid = 0
    candidate_evidence_total = 0
    case_results: list[dict[str, Any]] = []
    aggregate: dict[str, Counter[str]] = {
        "direct": Counter(),
        "dependency_linked": Counter(),
        "impact": Counter(),
    }
    context_labels = 0
    context_classified = 0
    context_false_impacts = 0
    marker_cases = 0
    marker_cases_passed = 0
    ambiguity_cases_passed = 0

    for case in labels:
        report = workflow.analyze(case["change"])
        direct = {
            _candidate_id(item)
            for item in report.affected_items
            if item.classification == "direct_target"
        }
        dependencies = {
            _candidate_id(item)
            for item in report.affected_items
            if item.classification == "dependency_linked"
        }
        impacts = direct | dependencies
        contexts = {_candidate_id(item) for item in report.context_items}
        expected_direct = set(case["expected_direct"])
        expected_dependencies = set(case["expected_dependency_linked"])
        expected_impacts = expected_direct | expected_dependencies
        expected_context = set(case["expected_context_only"])
        _update_counts(aggregate["direct"], expected_direct, direct)
        _update_counts(aggregate["dependency_linked"], expected_dependencies, dependencies)
        _update_counts(aggregate["impact"], expected_impacts, impacts)
        context_labels += len(expected_context)
        context_classified += len(expected_context & contexts)
        context_false_impacts += len(expected_context & impacts)

        evidence_by_id = {item.evidence_id: item for item in report.evidence}
        for reference in report.evidence:
            source = source_by_path.get(reference.path)
            valid = bool(
                source
                and 1
                <= reference.start_line
                <= reference.end_line
                <= len(source.content.splitlines())
            )
            evidence_counts["total"] += 1
            evidence_counts["valid"] += int(valid)
        for candidate in report.affected_items:
            candidate_evidence_total += 1
            location_valid = any(
                evidence_id in evidence_by_id
                and evidence_by_id[evidence_id].kind == "source"
                and evidence_by_id[evidence_id].path == candidate.path
                and evidence_by_id[evidence_id].symbol == candidate.symbol
                for evidence_id in candidate.location_evidence_ids
            )
            relationship_valid = candidate.classification != "dependency_linked" or any(
                evidence_id in evidence_by_id and evidence_by_id[evidence_id].kind == "dependency"
                for evidence_id in candidate.relationship_evidence_ids
            )
            candidate_evidence_valid += int(location_valid and relationship_valid)

        markers = case["expected_test_markers"]
        if markers:
            marker_cases += 1
            suggestion_text = "\n".join(item.behavior for item in report.suggested_tests)
            marker_cases_passed += int(all(marker in suggestion_text for marker in markers))
        ambiguity_disclosed = any(
            "lacks an explicit expected result" in text for text in report.uncertainties
        )
        ambiguity_cases_passed += int(ambiguity_disclosed == case["expected_ambiguous"])
        case_results.append(
            {
                "id": case["id"],
                "direct_targets_match": direct == expected_direct,
                "dependency_candidates_match": dependencies == expected_dependencies,
                "context_false_impacts": sorted(expected_context & impacts),
                "test_idea_markers_match": not markers
                or all(
                    marker in "\n".join(item.behavior for item in report.suggested_tests)
                    for marker in markers
                ),
                "ambiguity_disclosed_as_labeled": ambiguity_disclosed == case["expected_ambiguous"],
            }
        )

    return {
        "method": method,
        "cases": len(labels),
        "direct_target_metrics": _prf(aggregate["direct"]),
        "dependency_candidate_metrics": _prf(aggregate["dependency_linked"]),
        "impact_candidate_metrics": _prf(aggregate["impact"]),
        "labeled_context_negatives": {
            "total": context_labels,
            "classified_as_context_only": context_classified,
            "incorrectly_classified_as_impact": context_false_impacts,
        },
        "labeled_context_false_impact_rate": _rate(context_false_impacts, context_labels),
        "test_idea_marker_coverage": _rate(marker_cases_passed, marker_cases),
        "ambiguity_disclosure_accuracy": _rate(ambiguity_cases_passed, len(labels)),
        "valid_source_reference_rate": _rate(evidence_counts["valid"], evidence_counts["total"]),
        "impact_candidate_evidence_support_rate": _rate(
            candidate_evidence_valid, candidate_evidence_total
        ),
        "case_results": case_results,
    }


def _candidate_id(candidate: Any) -> str:
    return f"{candidate.path}::{candidate.symbol}"


def _update_counts(counts: Counter[str], expected: set[str], actual: set[str]) -> None:
    counts["true_positive"] += len(expected & actual)
    counts["false_positive"] += len(actual - expected)
    counts["false_negative"] += len(expected - actual)


def _prf(counts: Counter[str]) -> dict[str, float | int]:
    true_positive = counts["true_positive"]
    false_positive = counts["false_positive"]
    false_negative = counts["false_negative"]
    precision = _rate(true_positive, true_positive + false_positive)
    recall = _rate(true_positive, true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
