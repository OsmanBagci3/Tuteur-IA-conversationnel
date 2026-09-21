"""Extraction PDF : texte page par page + images avec dédup et filtrage.

On utilise PyMuPDF (`fitz`) qui expose à la fois le texte et les images
embarquées dans le PDF via une API cohérente.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageText:
    """Texte extrait d'une page du PDF."""

    page: int  # numéro de page 1-indexé
    text: str


@dataclass(frozen=True)
class ExtractedImage:
    """Image extraite du PDF, sauvegardée sur disque."""

    page: int  # première page où l'image apparaît
    xref: int  # identifiant unique de l'image dans le PDF
    path: Path  # chemin du fichier image extrait
    width: int
    height: int


def extract_pdf(
    pdf_path: Path,
    images_dir: Path,
    min_image_width: int = 100,
    min_image_height: int = 100,
) -> tuple[list[PageText], list[ExtractedImage]]:
    """Extrait le texte page par page et les images uniques du PDF.

    Args:
        pdf_path: Chemin vers le PDF source.
        images_dir: Dossier de sortie où sauvegarder les images extraites.
        min_image_width: Largeur minimum (pixels) pour conserver une image.
        min_image_height: Hauteur minimum (pixels) pour conserver une image.

    Returns:
        Un tuple (pages_texte, images) avec :
            - pages_texte : une entrée par page contenant du texte non vide
            - images : liste dédupliquée d'images assez grandes
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

    images_dir.mkdir(parents=True, exist_ok=True)

    pages: list[PageText] = []
    images: list[ExtractedImage] = []
    seen_xrefs: set[int] = set()

    logger.info("Ouverture du PDF : %s", pdf_path)
    with fitz.open(pdf_path) as doc:
        logger.info("Pages détectées : %d", len(doc))

        for page_num, page in enumerate(doc, start=1):
            # --- Texte ---
            text = page.get_text("text").strip()
            if text:
                pages.append(PageText(page=page_num, text=text))

            # --- Images ---
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                try:
                    base_image = doc.extract_image(xref)
                except Exception as exc:  # image corrompue / non supportée
                    logger.warning("Image xref=%d ignorée (%s)", xref, exc)
                    continue

                image_bytes: bytes = base_image["image"]
                ext: str = base_image["ext"]  # png, jpeg, ...

                # Ouvre l'image pour vérifier ses dimensions.
                try:
                    with Image.open(_bytes_io(image_bytes)) as pil_img:
                        w, h = pil_img.size
                        mode = pil_img.mode
                        if w < min_image_width or h < min_image_height:
                            continue
                        # Convertit en RGB pour compatibilité CLIP (évite CMYK/L).
                        if mode != "RGB":
                            pil_img = pil_img.convert("RGB")
                            ext = "png"
                            out_path = images_dir / f"p{page_num:03d}_x{xref}.{ext}"
                            pil_img.save(out_path, format="PNG")
                        else:
                            out_path = images_dir / f"p{page_num:03d}_x{xref}.{ext}"
                            out_path.write_bytes(image_bytes)
                except Exception as exc:
                    logger.warning("Image xref=%d illisible (%s)", xref, exc)
                    continue

                images.append(
                    ExtractedImage(
                        page=page_num,
                        xref=xref,
                        path=out_path,
                        width=w,
                        height=h,
                    )
                )

    logger.info(
        "Extraction OK : %d pages avec texte, %d images conservées",
        len(pages),
        len(images),
    )
    return pages, images


def _bytes_io(data: bytes):
    """Import local pour éviter d'exposer io au niveau module."""
    from io import BytesIO

    return BytesIO(data)
