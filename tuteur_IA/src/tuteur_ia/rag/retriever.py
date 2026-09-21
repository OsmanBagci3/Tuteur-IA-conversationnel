"""Retriever hybride multimodal.

À partir d'une requête texte, interroge :
- la collection Chroma "text"  avec l'embedding sentence-transformers,
- la collection Chroma "image" avec l'embedding CLIP-text (aligné avec l'espace image).

Retourne deux listes ordonnées (chunks texte + images) et, en bonus,
un ranking unifié via RRF pour un affichage "top sources".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from tuteur_ia.ingestion.embedders import CLIPImageEmbedder, TextEmbedder
from tuteur_ia.ingestion.indexer import IMAGE_COLLECTION, TEXT_COLLECTION
from tuteur_ia.rag.fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedText:
    id: str
    text: str
    page: int
    chunk_index: int
    source: str
    score: float  # 1 - cosine_distance ∈ [-1, 1]


@dataclass(frozen=True)
class RetrievedImage:
    id: str
    path: Path
    page: int
    xref: int
    source: str
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    texts: list[RetrievedText]
    images: list[RetrievedImage]
    unified_ranking: list[tuple[str, float]]  # (id, rrf_score)


class HybridRetriever:
    """Retriever multimodal Chroma + Sentence-Transformers + CLIP."""

    def __init__(
        self,
        chroma_dir: Path,
        text_embedder: TextEmbedder,
        clip_embedder: CLIPImageEmbedder,
        rrf_k: int = 60,
    ) -> None:
        client = chromadb.PersistentClient(path=str(chroma_dir))
        self._text_col: Collection = client.get_collection(TEXT_COLLECTION)
        self._image_col: Collection = client.get_collection(IMAGE_COLLECTION)
        self._text_embedder = text_embedder
        self._clip_embedder = clip_embedder
        self._rrf_k = rrf_k
        logger.info(
            "Retriever prêt : %d chunks texte, %d images",
            self._text_col.count(),
            self._image_col.count(),
        )

    def retrieve(
        self,
        query: str,
        k_text: int = 5,
        k_image: int = 3,
    ) -> RetrievalResult:
        """Interroge les deux collections avec le même prompt."""
        texts = self._search_text(query, k_text) if k_text > 0 else []
        images = self._search_image(query, k_image) if k_image > 0 else []

        unified = reciprocal_rank_fusion(
            ranked_lists=[[t.id for t in texts], [i.id for i in images]],
            k=self._rrf_k,
        )
        return RetrievalResult(texts=texts, images=images, unified_ranking=unified)

    # ------------------------------------------------------------------ #
    # Recherche texte
    # ------------------------------------------------------------------ #
    def _search_text(self, query: str, k: int) -> list[RetrievedText]:
        embedding = self._text_embedder.encode([query])[0].tolist()
        raw = self._text_col.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedText(
                id=_first(raw["ids"], i),
                text=_first(raw["documents"], i),
                page=int(_first(raw["metadatas"], i).get("page", 0)),
                chunk_index=int(_first(raw["metadatas"], i).get("chunk_index", 0)),
                source=str(_first(raw["metadatas"], i).get("source", "")),
                score=1.0 - float(_first(raw["distances"], i)),
            )
            for i in range(len(raw["ids"][0]))
        ]

    # ------------------------------------------------------------------ #
    # Recherche image (via CLIP text encoder)
    # ------------------------------------------------------------------ #
    def _search_image(self, query: str, k: int) -> list[RetrievedImage]:
        embedding = self._clip_embedder.encode_text([query])[0].tolist()
        raw = self._image_col.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["metadatas", "distances"],
        )
        return [
            RetrievedImage(
                id=_first(raw["ids"], i),
                path=Path(str(_first(raw["metadatas"], i).get("path", ""))),
                page=int(_first(raw["metadatas"], i).get("page", 0)),
                xref=int(_first(raw["metadatas"], i).get("xref", 0)),
                source=str(_first(raw["metadatas"], i).get("source", "")),
                score=1.0 - float(_first(raw["distances"], i)),
            )
            for i in range(len(raw["ids"][0]))
        ]


def _first(field: list, i: int):
    """Chroma renvoie tout sous forme `[[...]]` (batch de 1) — on unwrap."""
    return field[0][i]
