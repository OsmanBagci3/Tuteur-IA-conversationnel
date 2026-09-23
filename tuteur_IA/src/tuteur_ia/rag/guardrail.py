"""Garde-fou anti-hallucination : décide si le sujet est couvert par le corpus.

Extrait de `scripts/query.py` (P2) pour être réutilisé par le moteur du tuteur (P3)
sans duplication.
"""

from __future__ import annotations

from tuteur_ia.rag.retriever import RetrievalResult


def is_topic_covered(retrieval: RetrievalResult, min_score: float) -> bool:
    """N'autorise l'appel LLM que si le meilleur score texte dépasse un seuil.

    Le score image n'est volontairement pas utilisé : il s'est révélé peu
    discriminant entre des questions sur-sujet et hors-sujet.
    """
    if not retrieval.texts:
        return False
    return retrieval.texts[0].score >= min_score
