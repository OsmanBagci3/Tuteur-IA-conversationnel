"""Ressources partagées de l'API : modèles et retriever chargés une seule fois
au démarrage (coûteux : embeddings + connexion Chroma), puis réutilisés par
toutes les sessions.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import settings
from tuteur_ia.ingestion.embedders import CLIPImageEmbedder, TextEmbedder
from tuteur_ia.llm.base import LLMClient
from tuteur_ia.llm.providers.mistral import MistralClient
from tuteur_ia.rag.retriever import HybridRetriever
from tuteur_ia.tutor.objectives import LearningPath, load_learning_path


@dataclass
class Resources:
    retriever: HybridRetriever
    llm: LLMClient
    learning_path: LearningPath


def build_resources() -> Resources:
    text_embedder = TextEmbedder(settings.text_embedding_model)
    clip_embedder = CLIPImageEmbedder(settings.clip_model_name, settings.clip_pretrained)
    retriever = HybridRetriever(
        chroma_dir=settings.chroma_dir,
        text_embedder=text_embedder,
        clip_embedder=clip_embedder,
        rrf_k=settings.rag_rrf_k,
    )
    llm = MistralClient(api_key=settings.mistral_api_key, model=settings.mistral_model)
    learning_path = load_learning_path(settings.learning_path_file)
    return Resources(retriever=retriever, llm=llm, learning_path=learning_path)
