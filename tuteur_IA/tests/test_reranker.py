from __future__ import annotations

from tuteur_ia.rag.reranker import CrossEncoderReranker
from tuteur_ia.rag.retriever import RetrievedText


class _FakePredictModel:
    """Remplace `sentence_transformers.CrossEncoder` : associe un score à chaque paire
    (requête, texte) selon le texte, pour tester la logique de réordonnancement sans
    charger de vrai modèle (lourd, réseau).
    """

    def __init__(self, scores_by_text: dict[str, float]) -> None:
        self._scores_by_text = scores_by_text

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self._scores_by_text[text] for _, text in pairs]


def _make_reranker(scores_by_text: dict[str, float]) -> CrossEncoderReranker:
    # Contourne `__init__` (qui charge un vrai CrossEncoder) pour injecter un modèle factice.
    reranker = object.__new__(CrossEncoderReranker)
    reranker._model = _FakePredictModel(scores_by_text)  # type: ignore[attr-defined]
    return reranker


def _text(id_: str, content: str, score: float = 0.5) -> RetrievedText:
    return RetrievedText(id=id_, text=content, page=1, chunk_index=0, source="test", score=score)


def test_rerank_reorders_by_cross_encoder_score():
    texts = [_text("a", "faible pertinence"), _text("b", "forte pertinence")]
    reranker = _make_reranker({"faible pertinence": 0.1, "forte pertinence": 0.9})
    result = reranker.rerank("query", texts)
    assert [t.id for t in result] == ["b", "a"]


def test_rerank_does_not_modify_bi_encoder_score():
    texts = [_text("a", "x", score=0.42), _text("b", "y", score=0.77)]
    reranker = _make_reranker({"x": 0.9, "y": 0.1})
    result = reranker.rerank("query", texts)
    scores_by_id = {t.id: t.score for t in result}
    assert scores_by_id == {"a": 0.42, "b": 0.77}


def test_rerank_single_or_empty_list_is_a_no_op():
    reranker = _make_reranker({})
    assert reranker.rerank("query", []) == []
    single = [_text("a", "x")]
    assert reranker.rerank("query", single) == single
