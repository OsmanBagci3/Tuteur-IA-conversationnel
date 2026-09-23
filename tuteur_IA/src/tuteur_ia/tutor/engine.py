"""Moteur du tuteur : orchestre auto-diagnostic, explication, quiz, évaluation et adaptation.

Ce module ne fait AUCUNE I/O (pas de `input()`/`print()`) : il expose des méthodes
retournant des dataclasses de "tour" (turn), pour rester réutilisable aussi bien
par un script CLI (P3) que par l'API FastAPI (P4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from config.settings import settings
from tuteur_ia.llm.base import LLMClient, Message
from tuteur_ia.rag.context import format_context
from tuteur_ia.rag.guardrail import is_topic_covered
from tuteur_ia.rag.retriever import HybridRetriever, RetrievalResult
from tuteur_ia.tutor import placement
from tuteur_ia.tutor.evaluator import EvaluationResult, evaluate_answer
from tuteur_ia.tutor.objectives import LearningPath, Objective
from tuteur_ia.tutor.quiz import generate_quiz_question
from tuteur_ia.tutor.state import Level, ObjectiveResult, Phase, SessionState

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "config" / "prompts"
EXPLAIN_PROMPT = _PROMPTS_DIR / "explain.txt"
QUIZ_PROMPT = _PROMPTS_DIR / "quiz.txt"
EVALUATE_PROMPT = _PROMPTS_DIR / "evaluate.txt"

_LEVEL_ORDER = [Level.BEGINNER, Level.INTERMEDIATE, Level.ADVANCED]


@dataclass(frozen=True)
class PlacementQuestionTurn:
    index: int
    total: int
    objective_title: str
    question: str


@dataclass(frozen=True)
class ExplainTurn:
    objective_title: str
    text: str
    source_pages: list[int]
    image_paths: list[Path]


@dataclass(frozen=True)
class QuizTurn:
    objective_title: str
    question: str


@dataclass(frozen=True)
class EvaluationTurn:
    score: int
    feedback: str
    level: Level
    level_changed: bool
    done: bool
    next_objective_title: str | None


class TutorEngine:
    """Machine à états pédagogique pour une session d'apprenant."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        learning_path: LearningPath,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._learning_path = learning_path
        self.state = SessionState()

        self._context_cache: dict[str, RetrievalResult] = {}
        self._pending_question: str | None = None
        # Liste de (objectif, question) en attente de réponse pendant le placement.
        self._placement_pending: list[tuple[Objective, str]] = []

    # ------------------------------------------------------------------ #
    # Auto-diagnostic initial
    # ------------------------------------------------------------------ #
    def start_placement(self) -> list[PlacementQuestionTurn]:
        """Génère les questions de placement (une par objectif-sonde)."""
        probes = placement.select_placement_objectives(self._learning_path.objectives)
        turns: list[PlacementQuestionTurn] = []
        self._placement_pending = []

        for i, objective in enumerate(probes):
            retrieval = self._get_retrieval(objective)
            context = format_context(retrieval)
            question = generate_quiz_question(
                self._llm,
                QUIZ_PROMPT,
                objective_title=objective.title,
                level=Level.INTERMEDIATE.value,
                context=context,
            )
            self._placement_pending.append((objective, question))
            turns.append(
                PlacementQuestionTurn(
                    index=i, total=len(probes), objective_title=objective.title, question=question
                )
            )
        return turns

    def submit_placement_answer(self, index: int, answer: str) -> None:
        """Note la réponse à la question de placement `index` et stocke le score."""
        objective, question = self._placement_pending[index]
        retrieval = self._get_retrieval(objective)
        context = format_context(retrieval)
        result = evaluate_answer(
            self._llm,
            EVALUATE_PROMPT,
            objective_title=objective.title,
            level=Level.INTERMEDIATE.value,
            question=question,
            answer=answer,
            context=context,
        )
        self.state.placement_scores.append(result.score)

    def finalize_placement(self) -> Level:
        """Calcule le niveau initial et démarre le parcours principal."""
        level = placement.initial_level_from_scores(self.state.placement_scores)
        self.state.level = level
        self.state.phase = Phase.EXPLAIN
        self.state.objective_index = 0
        return level

    # ------------------------------------------------------------------ #
    # Boucle principale : explain -> quiz -> evaluate -> adapt
    # ------------------------------------------------------------------ #
    def current_objective(self) -> Objective:
        return self._learning_path.objectives[self.state.objective_index]

    def is_done(self) -> bool:
        return self.state.phase == Phase.DONE

    def explain_current_objective(self) -> ExplainTurn:
        objective = self.current_objective()
        retrieval = self._get_retrieval(objective)

        if not is_topic_covered(retrieval, settings.rag_min_score):
            # Ne devrait pas arriver si le learning_path est bien aligné avec le
            # corpus, mais on log pour le signaler plutôt que d'échouer en silence.
            logger.warning(
                "Objectif '%s' : score de couverture faible, contenu potentiellement pauvre.",
                objective.id,
            )

        context = format_context(retrieval)
        template = EXPLAIN_PROMPT.read_text(encoding="utf-8")
        request = f"Explique-moi le concept suivant : {objective.title}. {objective.summary}".strip()
        prompt = template.format(context=context, level=self.state.level.value, question=request)

        image_paths = [img.path for img in retrieval.images]
        answer = self._llm.chat([Message(role="user", content=prompt, images=image_paths)])

        self.state.phase = Phase.QUIZ
        return ExplainTurn(
            objective_title=objective.title,
            text=answer,
            source_pages=sorted({t.page for t in retrieval.texts}),
            image_paths=image_paths,
        )

    def generate_quiz(self) -> QuizTurn:
        objective = self.current_objective()
        retrieval = self._get_retrieval(objective)
        context = format_context(retrieval)

        question = generate_quiz_question(
            self._llm,
            QUIZ_PROMPT,
            objective_title=objective.title,
            level=self.state.level.value,
            context=context,
        )
        self._pending_question = question
        return QuizTurn(objective_title=objective.title, question=question)

    def evaluate_answer(self, answer: str) -> EvaluationTurn:
        if self._pending_question is None:
            raise RuntimeError("Aucune question de quiz en attente — appelle generate_quiz() d'abord.")

        objective = self.current_objective()
        retrieval = self._get_retrieval(objective)
        context = format_context(retrieval)

        result: EvaluationResult = evaluate_answer(
            self._llm,
            EVALUATE_PROMPT,
            objective_title=objective.title,
            level=self.state.level.value,
            question=self._pending_question,
            answer=answer,
            context=context,
        )
        self._pending_question = None

        old_level = self.state.level
        new_level = self._adapt_level(old_level, result.score)
        self.state.level = new_level
        self.state.results.append(
            ObjectiveResult(objective_id=objective.id, score=result.score, level_at_time=old_level)
        )

        self.state.objective_index += 1
        done = self.state.objective_index >= len(self._learning_path.objectives)
        next_title: str | None = None
        if done:
            self.state.phase = Phase.DONE
        else:
            self.state.phase = Phase.EXPLAIN
            next_title = self.current_objective().title

        return EvaluationTurn(
            score=result.score,
            feedback=result.feedback,
            level=new_level,
            level_changed=new_level != old_level,
            done=done,
            next_objective_title=next_title,
        )

    # ------------------------------------------------------------------ #
    # Internes
    # ------------------------------------------------------------------ #
    def _get_retrieval(self, objective: Objective) -> RetrievalResult:
        """Récupère (et met en cache) le contexte RAG d'un objectif, réutilisé par
        explain/quiz/evaluate pour rester cohérent sur tout le cycle de l'objectif.
        """
        if objective.id not in self._context_cache:
            self._context_cache[objective.id] = self._retriever.retrieve(
                query=objective.query_hint,
                k_text=settings.rag_top_k_text,
                k_image=settings.rag_top_k_image,
            )
        return self._context_cache[objective.id]

    @staticmethod
    def _adapt_level(current: Level, score: int) -> Level:
        idx = _LEVEL_ORDER.index(current)
        if score >= settings.adapt_score_up and idx < len(_LEVEL_ORDER) - 1:
            return _LEVEL_ORDER[idx + 1]
        if score <= settings.adapt_score_down and idx > 0:
            return _LEVEL_ORDER[idx - 1]
        return current
