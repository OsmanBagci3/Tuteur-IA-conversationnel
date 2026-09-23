"""Mise en forme d'un `RetrievalResult` en bloc de contexte textuel pour un prompt."""

from __future__ import annotations

from tuteur_ia.rag.retriever import RetrievalResult


def format_context(retrieval: RetrievalResult) -> str:
    """Construit le bloc CONTEXTE injecté dans les prompts (explain/quiz/evaluate)."""
    blocks: list[str] = []
    for i, t in enumerate(retrieval.texts, start=1):
        blocks.append(f"[Extrait {i} — page {t.page}]\n{t.text.strip()}")
    if retrieval.images:
        pages = sorted({img.page for img in retrieval.images})
        blocks.append(f"[Schémas joints depuis les pages : {', '.join(map(str, pages))}]")
    return "\n\n".join(blocks) if blocks else "(aucun extrait pertinent trouvé)"
