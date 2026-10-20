from dataclasses import dataclass

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
