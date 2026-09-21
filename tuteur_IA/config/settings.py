"""Settings globales du projet, chargées depuis `.env` avec validation Pydantic."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Racine du projet = dossier `tuteur_IA/` (2 niveaux au-dessus de ce fichier).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Configuration typée du projet.

    Les valeurs sont lues depuis `.env` puis surchargées par l'environnement.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM ---
    mistral_api_key: str = Field(default="", description="Clé API Mistral.")
    mistral_model: str = Field(default="pixtral-12b-2409")

    # --- Embeddings ---
    text_embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    clip_model_name: str = Field(default="ViT-B-32")
    clip_pretrained: str = Field(default="laion2b_s34b_b79k")

    # --- RAG ---
    rag_top_k_text: int = Field(default=5, ge=1, le=20)
    rag_top_k_image: int = Field(default=3, ge=0, le=20)
    rag_rrf_k: int = Field(default=60, ge=1)

    # --- Chemins (relatifs à PROJECT_ROOT) ---
    corpus_path: str = Field(default="data/raw/Natural Language Processing-1.pdf")
    extracted_images_dir: str = Field(default="data/extracted/images")
    chroma_persist_dir: str = Field(default="data/chroma")

    # --- API / UI ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    ui_port: int = Field(default=8501)
    api_base_url: str = Field(default="http://localhost:8000")

    # --- Helpers pour manipuler les chemins comme des Path absolus ---
    @property
    def corpus_file(self) -> Path:
        return (PROJECT_ROOT / self.corpus_path).resolve()

    @property
    def images_dir(self) -> Path:
        return (PROJECT_ROOT / self.extracted_images_dir).resolve()

    @property
    def chroma_dir(self) -> Path:
        return (PROJECT_ROOT / self.chroma_persist_dir).resolve()


settings = Settings()
