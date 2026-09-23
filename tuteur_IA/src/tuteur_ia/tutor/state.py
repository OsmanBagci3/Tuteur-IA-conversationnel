"""État de session du tuteur : niveau de l'apprenant, phase courante, progression."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Level(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class Phase(str, Enum):
    PLACEMENT = "placement"
    EXPLAIN = "explain"
    QUIZ = "quiz"
    DONE = "done"


@dataclass(frozen=True)
class ObjectiveResult:
    """Résultat d'un objectif complété (utile pour un futur affichage de progression)."""

    objective_id: str
    score: int  # 1-5
    level_at_time: Level


@dataclass
class SessionState:
    """État mutable d'une session de tuteur (un apprenant, un parcours)."""

    level: Level = Level.INTERMEDIATE  # niveau par défaut avant placement
    phase: Phase = Phase.PLACEMENT
    objective_index: int = 0
    results: list[ObjectiveResult] = field(default_factory=list)

    # Scores collectés pendant l'auto-diagnostic initial (avant qu'un niveau soit fixé).
    placement_scores: list[int] = field(default_factory=list)
