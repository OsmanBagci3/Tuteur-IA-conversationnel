"""Reranking optionnel des résultats texte via un cross-encoder (P6c).

Le bi-encoder (sentence-transformers) utilisé pour la recherche initiale encode
la requête et chaque document *indépendamment* (rapide, scalable à toute la base),
mais perd en précision par rapport à un cross-encoder qui encode la paire
(requête, document) conjointement. On l'utilise donc uniquement pour réordonner
les quelques candidats déjà récupérés par la recherche vectorielle (top-k), pas
pour la recherche elle-même — coût négligeable (quelques chunks), gain de
précision sur l'ordre final.

Important : le score du bi-encoder (`RetrievedText.score`), utilisé par le
garde-fou anti-hallucination (`rag/guardrail.py`), n'est jamais remplacé : seul
l'ORDRE de la liste est modifié par le reranker.
"""

from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

from tuteur_ia.rag.retriever import RetrievedText

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Réordonne une liste de chunks texte par pertinence vis-à-vis d'une requête."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model = CrossEncoder(model_name)
        logger.info("Cross-encoder reranker chargé (%s).", model_name)

    def rerank(self, query: str, texts: list[RetrievedText]) -> list[RetrievedText]:
        """Retourne `texts` réordonné par score de pertinence cross-encoder décroissant.

        Ne modifie jamais `texts[i].score` : seul l'ordre de la liste change.
        """
        if len(texts) <= 1:
            return texts
        pairs = [(query, t.text) for t in texts]
        ce_scores = self._model.predict(pairs)
        ranked = sorted(zip(texts, ce_scores), key=lambda pair: pair[1], reverse=True)
        return [t for t, _ in ranked]
