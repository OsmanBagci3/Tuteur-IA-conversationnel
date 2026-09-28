"""Séquenceur de conversation : transforme les tours "bruts" du `TutorEngine`
(placement / explain / quiz / evaluate) en une suite de tours de chat.

Aucune I/O ici (pas de HTTP, pas de `print`/`input`) : ce module est réutilisé
tel quel par l'API FastAPI (P4) et pourrait l'être par le CLI (P3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from tuteur_ia.tutor.engine import PlacementQuestionTurn, TutorEngine
from tuteur_ia.tutor.state import Phase

TurnKind = Literal["placement_question", "system", "explain", "quiz", "evaluation"]


@dataclass(frozen=True)
class ChatTurn:
    """Un message unitaire produit par le tuteur, affichable tel quel côté UI."""

    kind: TurnKind
    text: str
    source_pages: list[int] = field(default_factory=list)
    image_paths: list[Path] = field(default_factory=list)
    score: int | None = None


class ChatSession:
    """Enveloppe un `TutorEngine` et pilote l'enchaînement placement -> parcours.

    `start()` et `send()` renvoient chacun une liste de `ChatTurn` (un tour
    utilisateur peut déclencher plusieurs messages tuteur d'affilée, ex. un
    feedback d'évaluation suivi immédiatement de l'explication de l'objectif
    suivant).
    """

    def __init__(self, engine: TutorEngine) -> None:
        self.engine = engine
        self._placement_turns: list[PlacementQuestionTurn] = []
        self._placement_cursor = 0

    def start(self) -> list[ChatTurn]:
        self._placement_turns = self.engine.start_placement()
        first = self._placement_turns[0]
        return [ChatTurn(kind="placement_question", text=first.question)]

    def send(self, content: str) -> list[ChatTurn]:
        if self.engine.state.phase == Phase.PLACEMENT:
            return self._handle_placement_answer(content)
        if self.engine.state.phase == Phase.QUIZ:
            return self._handle_quiz_answer(content)
        return [ChatTurn(kind="system", text="Le parcours est terminé, il n'y a plus rien à répondre.")]

    # ------------------------------------------------------------------ #
    # Internes
    # ------------------------------------------------------------------ #
    def _handle_placement_answer(self, content: str) -> list[ChatTurn]:
        current = self._placement_turns[self._placement_cursor]
        self.engine.submit_placement_answer(current.index, content)
        self._placement_cursor += 1

        if self._placement_cursor < len(self._placement_turns):
            next_turn = self._placement_turns[self._placement_cursor]
            return [ChatTurn(kind="placement_question", text=next_turn.question)]

        level = self.engine.finalize_placement()
        turns = [ChatTurn(kind="system", text=f"Niveau initial estimé : {level.value}.")]
        turns.extend(self._start_current_objective())
        return turns

    def _handle_quiz_answer(self, content: str) -> list[ChatTurn]:
        result = self.engine.evaluate_answer(content)
        turns = [ChatTurn(kind="evaluation", text=result.feedback, score=result.score)]

        if result.level_changed:
            turns.append(ChatTurn(kind="system", text=f"Niveau ajusté : {result.level.value}."))

        if result.done:
            turns.append(ChatTurn(kind="system", text="Parcours terminé, bravo !"))
        else:
            turns.extend(self._start_current_objective())
        return turns

    def _start_current_objective(self) -> list[ChatTurn]:
        explain_turn = self.engine.explain_current_objective()
        quiz_turn = self.engine.generate_quiz()
        return [
            ChatTurn(
                kind="explain",
                text=explain_turn.text,
                source_pages=explain_turn.source_pages,
                image_paths=explain_turn.image_paths,
            ),
            ChatTurn(kind="quiz", text=quiz_turn.question),
        ]
