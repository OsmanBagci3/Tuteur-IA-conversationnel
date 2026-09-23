"""Chargement du parcours pédagogique (`config/learning_path.yaml`)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Objective:
    """Un objectif pédagogique du parcours."""

    id: str
    title: str
    query_hint: str
    summary: str


@dataclass(frozen=True)
class LearningPath:
    subject: str
    levels: list[str]
    objectives: list[Objective]


def load_learning_path(path: Path) -> LearningPath:
    """Charge et valide le parcours pédagogique depuis un fichier YAML."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    objectives = [
        Objective(
            id=o["id"],
            title=o["title"],
            query_hint=o["query_hint"],
            summary=o.get("summary", ""),
        )
        for o in raw.get("objectives", [])
    ]
    if not objectives:
        raise ValueError(f"Aucun objectif défini dans {path}.")
    return LearningPath(
        subject=raw["subject"],
        levels=list(raw["levels"]),
        objectives=objectives,
    )
