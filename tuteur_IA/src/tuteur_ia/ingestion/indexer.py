"""Indexation des chunks texte et des images dans deux collections Chroma.

Chroma est utilisé en mode `PersistentClient` (fichiers locaux) : zéro
infrastructure requise pour la V1. Les embeddings sont **pré-calculés**
puis fournis directement à Chroma — la collection n'a donc pas besoin
d'une `embedding_function`, ce qui rend le pipeline entièrement contrôlable
et rejouable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection

from tuteur_ia.ingestion.chunker import TextChunk
from tuteur_ia.ingestion.loader import ExtractedImage

logger = logging.getLogger(__name__)

TEXT_COLLECTION = "text"
IMAGE_COLLECTION = "image"


class ChromaIndexer:
    """Wrapper autour d'un client Chroma persistant avec 2 collections."""

    def __init__(self, persist_dir: Path) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Chroma persistant : %s", persist_dir)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        # `hnsw:space=cosine` car nos embeddings sont L2-normalisés.
        self._text = self._client.get_or_create_collection(
            name=TEXT_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._image = self._client.get_or_create_collection(
            name=IMAGE_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    # --- Accès bas niveau (utilisé aussi par le retriever) ---
    @property
    def text_collection(self) -> Collection:
        return self._text

    @property
    def image_collection(self) -> Collection:
        return self._image

    # --- Reset ---
    def reset(self) -> None:
        """Purge les 2 collections. À utiliser avant re-ingestion complète."""
        logger.warning("Purge des collections Chroma (reset).")
        self._client.delete_collection(TEXT_COLLECTION)
        self._client.delete_collection(IMAGE_COLLECTION)
        self._text = self._client.get_or_create_collection(
            name=TEXT_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        self._image = self._client.get_or_create_collection(
            name=IMAGE_COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    # --- Insertion texte ---
    def add_text_chunks(
        self,
        chunks: list[TextChunk],
        embeddings: np.ndarray,
        source: str,
        batch_size: int = 256,
    ) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Nombre de chunks et d'embeddings incohérent.")
        ids = [f"text-p{c.page:03d}-c{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {"page": c.page, "chunk_index": c.chunk_index, "source": source}
            for c in chunks
        ]
        self._batched_upsert(self._text, ids, embeddings, documents, metadatas, batch_size)
        logger.info("Indexé %d chunks texte.", len(chunks))

    # --- Insertion images ---
    def add_images(
        self,
        images: list[ExtractedImage],
        embeddings: np.ndarray,
        source: str,
        batch_size: int = 128,
    ) -> None:
        if len(images) != embeddings.shape[0]:
            raise ValueError("Nombre d'images et d'embeddings incohérent.")
        ids = [f"img-p{img.page:03d}-x{img.xref}" for img in images]
        # Le "document" côté image est un placeholder textuel (page + fichier)
        # qui reste utile pour le débogage et pour l'affichage rapide.
        documents = [f"page {img.page} — {img.path.name}" for img in images]
        metadatas = [
            {
                "page": img.page,
                "xref": img.xref,
                "path": str(img.path),
                "width": img.width,
                "height": img.height,
                "source": source,
            }
            for img in images
        ]
        self._batched_upsert(self._image, ids, embeddings, documents, metadatas, batch_size)
        logger.info("Indexé %d images.", len(images))

    # --- Interne : upsert par batch ---
    @staticmethod
    def _batched_upsert(
        collection: Collection,
        ids: list[str],
        embeddings: np.ndarray,
        documents: list[str],
        metadatas: list[dict],
        batch_size: int,
    ) -> None:
        n = len(ids)
        for start in range(0, n, batch_size):
            end = start + batch_size
            collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end].tolist(),
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
