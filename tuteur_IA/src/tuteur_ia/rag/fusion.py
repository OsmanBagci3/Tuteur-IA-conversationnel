"""Reciprocal Rank Fusion (RRF).

Combine plusieurs listes ordonnées d'identifiants en un seul ranking en
sommant `1 / (k + rank)` sur toutes les listes où un identifiant apparaît.

Référence : Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods" (2009). La constante `k=60` est le
défaut recommandé par le papier ; elle amortit l'influence des rangs
très hauts.
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fusionne plusieurs listes ordonnées d'IDs et retourne les IDs triés par score.

    Args:
        ranked_lists: Chaque liste est ordonnée du plus pertinent au moins pertinent.
        k: Constante d'amortissement RRF (60 est le défaut classique).

    Returns:
        Liste `[(id, score), ...]` triée par score décroissant.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
