"""Encodage des chunks texte et des images en vecteurs normalisés.

- Texte  : `sentence-transformers` (modèle configurable).
- Images : `open_clip` (ViT-B-32 par défaut). Expose aussi l'encodage de
  requêtes textuelles pour permettre la recherche `texte -> image` au
  moment du retrieval.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class TextEmbedder:
    """Wrapper léger autour de sentence-transformers avec normalisation L2."""

    def __init__(self, model_name: str) -> None:
        # Import différé : évite de charger torch si on n'utilise que CLIP.
        from sentence_transformers import SentenceTransformer

        logger.info("Chargement du modèle texte : %s", model_name)
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Retourne une matrice (n, dim) de vecteurs normalisés."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)


class CLIPImageEmbedder:
    """Encode images ET requêtes texte dans l'espace CLIP partagé."""

    def __init__(self, model_name: str, pretrained: str) -> None:
        import open_clip

        logger.info("Chargement CLIP : %s / %s", model_name, pretrained)
        self.model_name = model_name
        self.pretrained = pretrained
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self._device
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(model_name)

    @property
    def dimension(self) -> int:
        # CLIP ViT-B-32 = 512 ; on lit dynamiquement pour éviter le hardcode.
        return int(self._model.visual.output_dim)

    def encode_images(self, image_paths: list[Path], batch_size: int = 16) -> np.ndarray:
        """Encode une liste d'images (par chemin) en vecteurs L2-normalisés."""
        if not image_paths:
            return np.zeros((0, self.dimension), dtype=np.float32)

        vectors: list[np.ndarray] = []
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            tensors: list[torch.Tensor] = []
            for path in batch_paths:
                with Image.open(path) as img:
                    tensors.append(self._preprocess(img.convert("RGB")))
            batch = torch.stack(tensors).to(self._device)
            with torch.no_grad():
                feats = self._model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            vectors.append(feats.cpu().numpy().astype(np.float32))
        return np.concatenate(vectors, axis=0)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Encode des requêtes textuelles dans le MÊME espace que les images."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        tokens = self._tokenizer(texts).to(self._device)
        with torch.no_grad():
            feats = self._model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)
