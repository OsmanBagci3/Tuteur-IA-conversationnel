"""Génération d'une question de vérification de compréhension via le LLM."""

from __future__ import annotations

from pathlib import Path

from tuteur_ia.llm.base import LLMClient, Message


def generate_quiz_question(
    llm: LLMClient,
    prompt_path: Path,
    objective_title: str,
    level: str,
    context: str,
) -> str:
    """Génère UNE question ouverte adaptée au niveau, ancrée dans le contexte fourni."""
    template = prompt_path.read_text(encoding="utf-8")
    prompt = template.format(objective_title=objective_title, level=level, context=context)
    raw = llm.chat([Message(role="user", content=prompt)], temperature=0.5)
    return raw.strip()
