"""CLI de test end-to-end pour la P2 : question -> RAG multimodal -> Mistral.

Usage :
    uv run python scripts/query.py "Qu'est-ce que le NLP ?"
    uv run python scripts/query.py "Explique-moi les embeddings" --show-images
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.settings import settings  # noqa: E402
from tuteur_ia.ingestion.embedders import CLIPImageEmbedder, TextEmbedder  # noqa: E402
from tuteur_ia.llm.base import Message  # noqa: E402
from tuteur_ia.llm.providers.mistral import MistralClient  # noqa: E402
from tuteur_ia.rag.retriever import HybridRetriever, RetrievalResult  # noqa: E402

PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "explain.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test end-to-end du RAG + Mistral.")
    parser.add_argument("question", help="La question à poser au tuteur.")
    parser.add_argument("--k-text", type=int, default=settings.rag_top_k_text)
    parser.add_argument("--k-image", type=int, default=settings.rag_top_k_image)
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Ne pas envoyer les images à Mistral (retrieval texte pur).",
    )
    return parser.parse_args()


def build_prompt(question: str, retrieval: RetrievalResult) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    context_blocks: list[str] = []
    for i, t in enumerate(retrieval.texts, start=1):
        context_blocks.append(
            f"[Extrait {i} — page {t.page}]\n{t.text.strip()}"
        )
    if retrieval.images:
        pages = sorted({img.page for img in retrieval.images})
        context_blocks.append(
            f"[Schémas joints depuis les pages : {', '.join(map(str, pages))}]"
        )
    context = "\n\n".join(context_blocks) if context_blocks else "(aucun extrait pertinent trouvé)"
    return template.format(context=context, question=question)


def print_sources(retrieval: RetrievalResult) -> None:
    print("\n--- Sources texte utilisées ---")
    for t in retrieval.texts:
        preview = t.text.replace("\n", " ")[:120]
        print(f"  • page {t.page} (score={t.score:.3f}) — {preview}...")
    if retrieval.images:
        print("\n--- Sources image utilisées ---")
        for img in retrieval.images:
            print(f"  • page {img.page} (score={img.score:.3f}) — {img.path}")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("query")
    args = parse_args()

    log.info("Chargement des encodeurs...")
    text_embedder = TextEmbedder(settings.text_embedding_model)
    clip_embedder = CLIPImageEmbedder(settings.clip_model_name, settings.clip_pretrained)

    log.info("Chargement du retriever...")
    retriever = HybridRetriever(
        chroma_dir=settings.chroma_dir,
        text_embedder=text_embedder,
        clip_embedder=clip_embedder,
        rrf_k=settings.rag_rrf_k,
    )

    log.info("Recherche pour : %r", args.question)
    result = retriever.retrieve(
        query=args.question,
        k_text=args.k_text,
        k_image=0 if args.no_images else args.k_image,
    )
    print_sources(result)

    log.info("Appel Mistral (%s)...", settings.mistral_model)
    client = MistralClient(api_key=settings.mistral_api_key, model=settings.mistral_model)
    prompt = build_prompt(args.question, result)

    image_paths = [img.path for img in result.images] if not args.no_images else []
    user_msg = Message(role="user", content=prompt, images=image_paths)
    answer = client.chat([user_msg])

    print("\n=== RÉPONSE DU TUTEUR ===\n")
    print(answer)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
