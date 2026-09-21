"""Abstraction LLM : interface `LLMClient` + types de messages agnostiques.

L'objectif est de pouvoir remplacer Mistral par un autre fournisseur
(OpenAI, Anthropic, Ollama...) sans modifier le code métier (retriever,
tutor engine, api). Chaque provider convertit la dataclass `Message`
vers son format natif au moment de l'appel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    """Message dans une conversation.

    - `content` : texte principal du message.
    - `images`  : chemins d'images locales à joindre (uniquement rôle "user"
      et pour des modèles multimodaux).
    """

    role: Role
    content: str
    images: list[Path] = field(default_factory=list)


class LLMClient(ABC):
    """Contrat minimal que tous les providers doivent implémenter."""

    model: str

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        """Génère une réponse à partir de l'historique de messages."""
