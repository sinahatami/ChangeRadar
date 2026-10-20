from changeradar.models import Analysis, DependencyEdge, SourceFile
from changeradar.parser import parse_source
from changeradar.retrieval import CodeChunk, CodeRetriever, build_code_chunks


class FakeEncoder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([1.0, 0.0] if "fruit" in lowered or "apple" in lowered else [0.0, 1.0])
        return vectors


class BrokenEncoder:
    def encode(self, texts):
        raise RuntimeError("model weights are unavailable")


def test_chunks_keep_original_source_and_symbol_line_range():
    source = SourceFile("sample.py", "def find_user(user_id):\n    return user_id\n")
    parsed = parse_source(source)
    chunks = build_code_chunks([source], [parsed])
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == "sample.py::find_user"
    assert chunk.path == "sample.py"
    assert (chunk.symbol_name, chunk.start_line, chunk.end_line) == ("find_user", 1, 2)
    assert chunk.text == source.content


def test_lexical_retrieval_matches_identifiers_and_exposes_scores():
    retriever = CodeRetriever(
        [
            _chunk("users.py", "find_user", "def find_user(user_id): return user_id"),
            _chunk("billing.py", "calculate_tax", "def calculate_tax(amount): return amount"),
        ]
    )
    response = retriever.retrieve_lexical("find user by user_id", top_k=1)
    assert response.hits[0].chunk.symbol_name == "find_user"
    assert response.hits[0].lexical_score > 0
    assert response.hits[0].score > 0


def test_semantic_retrieval_uses_injected_encoder_without_model_download():
    retriever = CodeRetriever(
        [
            _chunk("fruit.py", "apple_handler", "def apple_handler(): return 'fruit'"),
            _chunk("vehicle.py", "car_handler", "def car_handler(): return 'vehicle'"),
        ],
        encoder=FakeEncoder(),
    )
    response = retriever.retrieve_semantic("fruit")
    assert response.effective_method == "semantic"
    assert response.hits[0].chunk.symbol_name == "apple_handler"
    assert response.hits[0].semantic_score > response.hits[1].semantic_score


def test_semantic_unavailable_falls_back_to_lexical():
    retriever = CodeRetriever(
        [_chunk("user.py", "find_user", "def find_user(): pass")], encoder=BrokenEncoder()
    )
    response = retriever.retrieve_semantic("find user")
    assert response.effective_method == "lexical_fallback"
    assert response.warning and "using lexical scores" in response.warning
    assert response.hits[0].lexical_score > 0


def test_hybrid_can_boost_chunks_connected_to_changed_symbol():
    changed = _chunk("service.py", "update_user", "def update_user(): pass")
    related = _chunk("store.py", "save_user", "def save_user(): pass")
    unrelated = _chunk("other.py", "unrelated", "def unrelated(): pass")
    analysis = Analysis(
        files=[],
        dependencies=[
            DependencyEdge("service:update_user", "store:save_user", "call", "service.py", 1)
        ],
    )
    retriever = CodeRetriever([changed, related, unrelated], analysis, encoder=FakeEncoder())
    response = retriever.retrieve_hybrid("no lexical matches", changed_symbols={"update_user"})
    boosted = next(hit for hit in response.hits if hit.chunk.symbol_name == "save_user")
    assert boosted.dependency_boost > 0
    assert boosted.score > 0


def _chunk(path, symbol, text):
    return CodeChunk(f"{path}::{symbol}", path, symbol, 1, 1, text)
