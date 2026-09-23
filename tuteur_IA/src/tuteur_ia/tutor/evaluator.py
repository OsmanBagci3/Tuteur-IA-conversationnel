"""Évaluation de la réponse d'un apprenant via le LLM, avec parsing structuré.

Le prompt `evaluate.txt` impose un format `SCORE: <1-5>` / `FEEDBACK: <texte>`
strict pour permettre un parsing fiable sans dépendre du JSON (plus robuste
aux variations de mise en forme d'un LLM).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from tuteur_ia.llm.base import LLMClient, Message

logger = logging.getLogger(__name__)

_SCORE_RE = re.compile(r"SCORE:\s*([1-5])", re.IGNORECASE)
_FEEDBACK_RE = re.compile(r"FEEDBACK:\s*(.+)", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class EvaluationResult:
    score: int  # 1-5
    feedback: str


def evaluate_answer(
    llm: LLMClient,
    prompt_path: Path,
    objective_title: str,
    level: str,
    question: str,
    answer: str,
    context: str,
) -> EvaluationResult:
    """Note la réponse de l'apprenant (1-5) avec un feedback court, sourcé sur le contexte."""
    template = prompt_path.read_text(encoding="utf-8")
    prompt = template.format(
        objective_title=objective_title,
        level=level,
        question=question,
        answer=answer,
        context=context,
    )
    raw = llm.chat([Message(role="user", content=prompt)], temperature=0.0)
    return _parse(raw)


def _parse(raw: str) -> EvaluationResult:
    score_match = _SCORE_RE.search(raw)
    feedback_match = _FEEDBACK_RE.search(raw)

    if not score_match:
        logger.warning("Format de score inattendu, fallback score=3. Réponse LLM: %r", raw)
        return EvaluationResult(score=3, feedback=raw.strip())

    score = int(score_match.group(1))
    feedback = feedback_match.group(1).strip() if feedback_match else raw.strip()
    return EvaluationResult(score=score, feedback=feedback)
