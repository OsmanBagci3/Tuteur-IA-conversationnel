"""Schémas Pydantic de l'API — contrat public, découplé des dataclasses internes."""

from __future__ import annotations

from pydantic import BaseModel


class MessageIn(BaseModel):
    session_id: str
    content: str


class ResetIn(BaseModel):
    session_id: str


class ChatTurnOut(BaseModel):
    kind: str
    text: str
    source_pages: list[int] = []
    image_urls: list[str] = []
    score: int | None = None


class SessionOut(BaseModel):
    session_id: str
    phase: str
    level: str
    objective_index: int
    total_objectives: int
    done: bool
    turns: list[ChatTurnOut]
