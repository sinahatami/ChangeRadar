from dataclasses import dataclass

from changeradar.evaluation import load_impact_labels, run_impact_evaluation
from scripts.evaluate_retrieval import metrics


@dataclass
class FakeChunk:
    chunk_id: str


@dataclass
class FakeHit:
    chunk: FakeChunk


@dataclass
class FakeResponse:
    hits: list[FakeHit]
    effective_method: str = "lexical"
    warning: str | None = None


class FakeRetriever:
    def retrieve(self, query, method, top_k):
        order = {
            "first": ["noise", "wanted"],
            "second": ["wanted", "noise"],
        }[query]
        return FakeResponse([FakeHit(FakeChunk(item)) for item in order[:top_k]])


def test_metrics_are_computed_from_ranked_hits():
    cases = [
        {"query": "first", "relevant": ["wanted"]},
        {"query": "second", "relevant": ["wanted"]},
    ]
    result = metrics(FakeRetriever(), cases, "lexical")
    assert result["Recall@5"] == 1.0
    assert result["MRR"] == 0.75


def test_labeled_impact_evaluation_runs_actual_offline_reports():
    labels = load_impact_labels()
    result = run_impact_evaluation(method="lexical", cases=labels)

    assert result["cases"] == 3
    assert result["direct_target_metrics"]["true_positive"] == 3
    assert result["direct_target_metrics"]["false_positive"] == 0
    assert result["dependency_candidate_metrics"]["true_positive"] == 1
    assert result["impact_candidate_evidence_support_rate"] == 1.0
    assert result["valid_source_reference_rate"] == 1.0
    assert result["labeled_context_false_impact_rate"] == 0.0
    assert result["test_idea_marker_coverage"] == 1.0
    assert result["ambiguity_disclosure_accuracy"] == 1.0
    assert all(
        case["direct_targets_match"]
        and case["dependency_candidates_match"]
        and case["test_idea_markers_match"]
        for case in result["case_results"]
    )
