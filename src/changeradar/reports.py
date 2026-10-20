"""Evidence-linked report records and citation validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidenceReference:
    evidence_id: str
    path: str
    symbol: str
    start_line: int
    end_line: int
    text: str
    kind: str = "source"


@dataclass
class ImpactCandidate:
    path: str
    symbol: str
    explanation: str
    evidence_ids: list[str]
    retrieval_score: float | None = None
    unsupported: bool = False


@dataclass
class RegressionTestSuggestion:
    title: str
    behavior: str
    evidence_ids: list[str]
    unsupported: bool = False


@dataclass
class ImpactReport:
    proposed_change: str
    summary: str
    summary_evidence_ids: list[str]
    affected_items: list[ImpactCandidate] = field(default_factory=list)
    suggested_tests: list[RegressionTestSuggestion] = field(default_factory=list)
    evidence: list[EvidenceReference] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    mode: str = "deterministic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def valid_evidence_ids(ids: Any, evidence: list[EvidenceReference]) -> tuple[list[str], list[str]]:
    """Return known citations and unknown IDs without trusting model-supplied locations."""
    if not isinstance(ids, list):
        return [], ["evidence_ids must be a list"]
    known = {item.evidence_id for item in evidence}
    accepted: list[str] = []
    errors: list[str] = []
    for evidence_id in ids:
        if not isinstance(evidence_id, str) or evidence_id not in known:
            errors.append(f"Unsupported evidence citation: {evidence_id!r}")
        elif evidence_id not in accepted:
            accepted.append(evidence_id)
    if not accepted:
        errors.append("Claim has no valid evidence citation")
    return accepted, errors
