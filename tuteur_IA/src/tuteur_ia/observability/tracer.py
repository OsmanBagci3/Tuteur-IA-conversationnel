"""Traçabilité/observabilité (P6c) : journalise chaque interaction significative
(retrieval + appel LLM) dans un fichier JSONL, pour inspection ultérieure
(débogage de la qualité du retrieval, audit des réponses, mesure de latence).

Volontairement minimal et "best effort" : une erreur d'écriture du log ne doit
jamais faire planter une session de tutorat (le traçage est un bonus
d'observabilité, pas une dépendance critique).
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TraceLogger:
    """Écrit une ligne JSON par événement dans `log_path`."""

    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path

    def log(self, event: str, **fields: Any) -> None:
        """Ajoute une ligne `{"event": ..., "timestamp": ..., **fields}` au fichier JSONL."""
        record = {"event": event, "timestamp": time.time(), **fields}
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            logger.warning(
                "Échec d'écriture de la trace dans %s (ignoré).", self._log_path, exc_info=True
            )

    @contextmanager
    def timed(self, event: str, **fields: Any):
        """Context manager qui mesure la durée du bloc et l'ajoute au champ `latency_ms`."""
        start = time.perf_counter()
        try:
            yield
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000, 1)
            self.log(event, latency_ms=latency_ms, **fields)
