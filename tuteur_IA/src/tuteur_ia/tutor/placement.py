"""Auto-diagnostic initial : place l'apprenant sur un des 3 niveaux avant de démarrer
le parcours, à partir de son score sur quelques questions repères.
"""

from __future__ import annotations

from tuteur_ia.tutor.objectives import Objective
from tuteur_ia.tutor.state import Level


def select_placement_objectives(objectives: list[Objective], count: int = 3) -> list[Objective]:
    """Choisit `count` objectifs répartis sur le parcours (début / milieu / fin).

    Servent de "sondes" pour estimer le niveau initial sans avoir à tout parcourir.
    """
    n = len(objectives)
    if n <= count:
        return list(objectives)

    # Indices répartis uniformément, dédupliqués en conservant l'ordre.
    indices = sorted({round(i * (n - 1) / (count - 1)) for i in range(count)})
    return [objectives[i] for i in indices]


def initial_level_from_scores(scores: list[int]) -> Level:
    """Calcule le niveau initial à partir de la moyenne des scores (échelle 1-5).

    Seuils volontairement simples et documentés (cf. JOURNAL.md) :
    moyenne <= 2 -> débutant, >= 4 -> avancé, sinon intermédiaire.
    """
    if not scores:
        return Level.INTERMEDIATE
    average = sum(scores) / len(scores)
    if average <= 2:
        return Level.BEGINNER
    if average >= 4:
        return Level.ADVANCED
    return Level.INTERMEDIATE
