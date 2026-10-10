from __future__ import annotations

from tuteur_ia.rag.guardrail import is_topic_covered
from tuteur_ia.rag.retriever import RetrievalResult, RetrievedText


def _result(*scores: float) -> RetrievalResult:
    texts = [
        RetrievedText(id=f"t{i}", text="texte", page=1, chunk_index=0, source="test", score=s)
        for i, s in enumerate(scores)
    ]
    return RetrievalResult(texts=texts, images=[], unified_ranking=[])


def test_covered_when_best_score_above_threshold():
    assert is_topic_covered(_result(0.5), min_score=0.25) is True


def test_not_covered_when_best_score_below_threshold():
    assert is_topic_covered(_result(0.1), min_score=0.25) is False


def test_covered_when_score_exactly_equals_threshold():
    assert is_topic_covered(_result(0.25), min_score=0.25) is True


def test_not_covered_when_no_text_results():
    assert is_topic_covered(_result(), min_score=0.25) is False


def test_only_the_best_score_matters():
    # Le meilleur score (0.9, premier élément) doit suffire même si un autre
    # extrait est très faible.
    assert is_topic_covered(_result(0.9, 0.01), min_score=0.25) is True


def test_image_score_is_never_used():
    # `is_topic_covered` ne doit regarder que `texts`, jamais `images` — on passe
    # une liste de textes vide, donc même avec une image très pertinente le sujet
    # est jugé non couvert.
    from pathlib import Path

    from tuteur_ia.rag.retriever import RetrievedImage

    result = RetrievalResult(
        texts=[],
        images=[
            RetrievedImage(id="i1", path=Path("x.png"), page=1, xref=1, source="test", score=0.99)
        ],
        unified_ranking=[],
    )
    assert is_topic_covered(result, min_score=0.25) is False
