"""Provider Mistral basé sur le SDK officiel `mistralai>=1.0`.

Gère les messages multimodaux : les images locales sont encodées en
data-URL base64 avant d'être envoyées à Pixtral.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

from mistralai.client import Mistral

from tuteur_ia.llm.base import LLMClient, Message

logger = logging.getLogger(__name__)

# Extensions autorisées par l'API vision de Mistral.
_ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "webp", "gif"}


class MistralClient(LLMClient):
    """Client Mistral (compatible modèles VLM comme Pixtral)."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY manquante. Renseigne-la dans .env avant d'appeler l'API."
            )
        self.model = model
        self._client = Mistral(api_key=api_key)

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        payload = [self._to_mistral_message(m) for m in messages]

        logger.info(
            "Mistral call: model=%s temp=%.2f messages=%d",
            self.model,
            temperature,
            len(payload),
        )
        response = self._client.chat.complete(
            model=self.model,
            messages=payload,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # `choices[0].message.content` peut être str ou list (multimodal),
        # mais l'assistant Mistral répond toujours en texte pur.
        content = response.choices[0].message.content  # type: ignore[union-attr]
        if not isinstance(content, str):
            content = "".join(
                getattr(chunk, "text", "") for chunk in content or []
            )
        return content.strip()

    # ------------------------------------------------------------------ #
    # Conversion Message -> format natif Mistral
    # ------------------------------------------------------------------ #
    def _to_mistral_message(self, msg: Message) -> dict[str, Any]:
        if not msg.images:
            return {"role": msg.role, "content": msg.content}

        if msg.role != "user":
            raise ValueError("Seuls les messages 'user' peuvent contenir des images.")

        parts: list[dict[str, Any]] = [{"type": "text", "text": msg.content}]
        for path in msg.images:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": _image_to_data_url(path),
                }
            )
        return {"role": "user", "content": parts}


def _image_to_data_url(path: Path) -> str:
    """Encode une image locale en data-URL base64 pour l'API Mistral."""
    ext = path.suffix.lstrip(".").lower()
    if ext == "jpg":
        ext = "jpeg"
    if ext not in _ALLOWED_IMAGE_EXT:
        raise ValueError(f"Extension image non supportée par Mistral : {path.suffix}")
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/{ext};base64,{data}"
