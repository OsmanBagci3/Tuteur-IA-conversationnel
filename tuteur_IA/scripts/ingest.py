"""Script CLI d'ingestion : PDF -> texte + images -> embeddings -> Chroma.

Usage :
    uv run python scripts/ingest.py                # ingestion (upsert idempotent)
    uv run python scripts/ingest.py --reset        # purge Chroma puis re-ingère
    uv run python scripts/ingest.py --sample 3     # limite à N pages (debug rapide)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Rend `config` et `src/tuteur_ia` importables sans installation editable préalable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.settings import settings  # noqa: E402
from tuteur_ia.ingestion.chunker import chunk_pages  # noqa: E402
from tuteur_ia.ingestion.embedders import CLIPImageEmbedder, TextEmbedder  # noqa: E402
from tuteur_ia.ingestion.indexer import ChromaIndexer  # noqa: E402
from tuteur_ia.ingestion.loader import extract_pdf  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingestion multimodale du corpus.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Purge les collections Chroma avant l'ingestion.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Ne traiter que les N premières pages (debug). 0 = tout.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("ingest")
    args = parse_args()

    corpus = settings.corpus_file
    if not corpus.exists():
        log.error("Corpus introuvable : %s", corpus)
        return 1

    # 1. Extraction PDF
    log.info("== 1/4 — Extraction PDF ==")
    pages, images = extract_pdf(
        pdf_path=corpus,
        images_dir=settings.images_dir,
        min_image_width=settings.min_image_width,
        min_image_height=settings.min_image_height,
    )
    if args.sample > 0:
        pages = pages[: args.sample]
        images = [img for img in images if img.page <= args.sample]
        log.info("Sample: %d pages, %d images retenues.", len(pages), len(images))

    # 2. Chunking
    log.info("== 2/4 — Chunking ==")
    chunks = chunk_pages(
        pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    log.info("Chunks produits : %d (à partir de %d pages)", len(chunks), len(pages))

    # 3. Embeddings
    log.info("== 3/4 — Embeddings ==")
    text_embedder = TextEmbedder(settings.text_embedding_model)
    text_vecs = text_embedder.encode([c.text for c in chunks])
    log.info("Embeddings texte : shape=%s", text_vecs.shape)

    image_vecs = None
    if images:
        clip = CLIPImageEmbedder(settings.clip_model_name, settings.clip_pretrained)
        image_vecs = clip.encode_images([img.path for img in images])
        log.info("Embeddings image : shape=%s", image_vecs.shape)
    else:
        log.warning("Aucune image conservée après filtrage — étape image ignorée.")

    # 4. Indexation
    log.info("== 4/4 — Indexation Chroma ==")
    indexer = ChromaIndexer(settings.chroma_dir)
    if args.reset:
        indexer.reset()

    source_name = corpus.name
    indexer.add_text_chunks(chunks, text_vecs, source=source_name)
    if images and image_vecs is not None:
        indexer.add_images(images, image_vecs, source=source_name)

    log.info("== Ingestion terminée ==")
    log.info("  - Chunks texte indexés : %d", len(chunks))
    log.info("  - Images indexées      : %d", len(images))
    log.info("  - Persist dir          : %s", settings.chroma_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
