"""Local lexical, semantic, and hybrid retrieval over parsed Python symbols."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

from .graph import module_name
from .models import Analysis, ParsedFile, SourceFile

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+")


@dataclass(frozen=True)
class CodeChunk:
    chunk_id: str
    path: str
    symbol_name: str
    start_line: int
    end_line: int
    text: str


@dataclass(frozen=True)
class RetrievalHit:
    chunk: CodeChunk
    score: float
    lexical_score: float
    semantic_score: float
    dependency_boost: float


@dataclass(frozen=True)
class RetrievalResponse:
    hits: list[RetrievalHit]
    requested_method: str
    effective_method: str
    warning: str | None = None


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEncoder:
    """Lazy CPU adapter; model weights are loaded only when retrieval is requested."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


def build_code_chunks(sources: list[SourceFile], parsed_files: list[ParsedFile]) -> list[CodeChunk]:
    """Build one evidence chunk per parsed symbol using its exact source lines."""
    source_by_path = {source.path: source for source in sources}
    chunks: list[CodeChunk] = []
    for parsed in parsed_files:
        source = source_by_path.get(parsed.path)
        if source is None or parsed.error:
            continue
        lines = source.content.splitlines(keepends=True)
        for symbol in parsed.symbols:
            text = "".join(lines[symbol.start_line - 1 : symbol.end_line])
            chunks.append(
                CodeChunk(
                    chunk_id=f"{symbol.path}::{symbol.qualified_name}",
                    path=symbol.path,
                    symbol_name=symbol.qualified_name,
                    start_line=symbol.start_line,
                    end_line=symbol.end_line,
                    text=text,
                )
            )
    return chunks


def _tokens(text: str) -> list[str]:
    pieces = TOKEN_RE.findall(text)
    expanded: list[str] = []
    for piece in pieces:
        expanded.extend(part.lower() for part in re.split(r"(?<=[a-z0-9])(?=[A-Z])", piece) if part)
        expanded.extend(part.lower() for part in piece.split("_") if part)
    return expanded


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _scaled(values: list[float]) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    minimum = min(values)
    if math.isclose(maximum, minimum):
        return [1.0 if value > 0 else 0.0 for value in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


class CodeRetriever:
    """BM25-style lexical search with optional local semantic and graph signals."""

    def __init__(
        self,
        chunks: list[CodeChunk],
        analysis: Analysis | None = None,
        encoder: Encoder | None = None,
        dependency_weight: float = 0.1,
    ) -> None:
        self.chunks = chunks
        self.analysis = analysis
        self.encoder = encoder
        self.dependency_weight = dependency_weight
        self._documents = [
            _tokens(f"{chunk.path} {chunk.symbol_name} {chunk.text}") for chunk in chunks
        ]
        self._semantic_vectors: list[list[float]] | None = None

    def _lexical_scores(self, query: str) -> list[float]:
        query_terms = _tokens(query)
        if not query_terms or not self.chunks:
            return [0.0] * len(self.chunks)
        doc_count = len(self._documents)
        lengths = [len(document) for document in self._documents]
        average_length = sum(lengths) / max(doc_count, 1)
        document_frequency: dict[str, int] = {}
        for term in set(query_terms):
            document_frequency[term] = sum(term in set(document) for document in self._documents)
        scores: list[float] = []
        for chunk, document, length in zip(self.chunks, self._documents, lengths):
            frequencies: dict[str, int] = {}
            for term in document:
                frequencies[term] = frequencies.get(term, 0) + 1
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                df = document_frequency[term]
                inverse_frequency = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
                denominator = frequency + 1.2 * (1 - 0.75 + 0.75 * length / max(average_length, 1))
                score += inverse_frequency * frequency * 2.2 / denominator
            query_identifier = " ".join(query_terms)
            identifier_tokens = _tokens(chunk.symbol_name)
            if query_identifier and all(term in identifier_tokens for term in query_terms):
                score += 1.5
            scores.append(score)
        return scores

    def _semantic_scores(self, query: str) -> list[float]:
        if self.encoder is None:
            self.encoder = SentenceTransformerEncoder()
        if self._semantic_vectors is None:
            self._semantic_vectors = self.encoder.encode([chunk.text for chunk in self.chunks])
        query_vector = self.encoder.encode([query])[0]
        return [(_cosine(query_vector, vector) + 1) / 2 for vector in self._semantic_vectors]

    def _dependency_targets(self, anchors: set[str]) -> set[str]:
        if self.analysis is None or not anchors:
            return set()
        targets: set[str] = set()
        for edge in self.analysis.dependencies:
            for anchor in anchors:
                if edge.source == anchor or edge.source.endswith(f":{anchor}"):
                    targets.add(edge.target)
                if edge.target == anchor or edge.target.endswith(f":{anchor}"):
                    targets.add(edge.source)
        return targets

    def retrieve(
        self,
        query: str,
        method: str = "hybrid",
        top_k: int = 5,
        changed_symbols: set[str] | None = None,
    ) -> RetrievalResponse:
        if method not in {"lexical", "semantic", "hybrid"}:
            raise ValueError("method must be lexical, semantic, or hybrid")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        lexical = self._lexical_scores(query)
        semantic: list[float] = [0.0] * len(self.chunks)
        effective = method
        warning = None
        if method in {"semantic", "hybrid"}:
            try:
                semantic = self._semantic_scores(query)
            except Exception as exc:
                warning = f"Semantic retrieval unavailable; using lexical scores ({exc})"
                effective = "lexical_fallback"
        lexical_scaled = _scaled(lexical)
        semantic_scaled = _scaled(semantic)
        related = (
            self._dependency_targets(changed_symbols or set()) if method == "hybrid" else set()
        )
        hits: list[RetrievalHit] = []
        for index, chunk in enumerate(self.chunks):
            boost = 0.0
            identity = f"{module_name(chunk.path)}:{chunk.symbol_name}"
            short_identity = f"{module_name(chunk.path)}:{chunk.symbol_name.split('.')[-1]}"
            if identity in related or short_identity in related:
                boost = self.dependency_weight
            if effective == "lexical_fallback":
                score = lexical_scaled[index]
            elif method == "lexical":
                score = lexical_scaled[index]
            elif method == "semantic":
                score = semantic_scaled[index]
            else:
                score = 0.5 * lexical_scaled[index] + 0.5 * semantic_scaled[index] + boost
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=score,
                    lexical_score=lexical[index],
                    semantic_score=semantic[index],
                    dependency_boost=boost,
                )
            )
        hits.sort(key=lambda hit: (-hit.score, hit.chunk.path, hit.chunk.start_line))
        return RetrievalResponse(hits[:top_k], method, effective, warning)

    def retrieve_lexical(self, query: str, top_k: int = 5) -> RetrievalResponse:
        return self.retrieve(query, "lexical", top_k)

    def retrieve_semantic(self, query: str, top_k: int = 5) -> RetrievalResponse:
        return self.retrieve(query, "semantic", top_k)

    def retrieve_hybrid(
        self, query: str, top_k: int = 5, changed_symbols: set[str] | None = None
    ) -> RetrievalResponse:
        return self.retrieve(query, "hybrid", top_k, changed_symbols)
