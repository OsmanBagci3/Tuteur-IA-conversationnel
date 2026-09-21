"""Découpage du texte en chunks avec chevauchement.

Implémentation minimale d'un *recursive character text splitter* : on tente
de couper sur des séparateurs de plus en plus fins (paragraphe → phrase →
mot → caractère) pour préserver la cohérence sémantique des chunks.
"""

from __future__ import annotations

from dataclasses import dataclass

from tuteur_ia.ingestion.loader import PageText

# Séparateurs testés dans l'ordre de préférence.
_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


@dataclass(frozen=True)
class TextChunk:
    """Fragment de texte prêt à être indexé."""

    page: int
    chunk_index: int  # index du chunk au sein de la page
    text: str


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[TextChunk]:
    """Découpe les pages en chunks avec chevauchement."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap doit être strictement inférieur à chunk_size.")

    all_chunks: list[TextChunk] = []
    for page in pages:
        parts = _split_recursive(page.text, chunk_size, chunk_overlap)
        for idx, part in enumerate(parts):
            all_chunks.append(TextChunk(page=page.page, chunk_index=idx, text=part))
    return all_chunks


def _split_recursive(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Découpe récursivement en tentant les séparateurs du plus grossier au plus fin."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    for sep in _SEPARATORS:
        if sep == "":
            # Dernier recours : découpe brute par taille de fenêtre.
            return _sliding_window(text, chunk_size, chunk_overlap)
        if sep in text:
            splits = text.split(sep)
            return _merge_splits(splits, sep, chunk_size, chunk_overlap)
    return _sliding_window(text, chunk_size, chunk_overlap)


def _merge_splits(
    splits: list[str],
    separator: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Recolle les splits en chunks proches de chunk_size, en gérant l'overlap."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    sep_len = len(separator)

    for piece in splits:
        piece = piece.strip()
        if not piece:
            continue
        # Si un seul morceau dépasse déjà la taille cible, on le re-splitte.
        if len(piece) > chunk_size:
            if current:
                chunks.append(separator.join(current).strip())
                current, current_len = [], 0
            chunks.extend(_split_recursive(piece, chunk_size, chunk_overlap))
            continue

        added_len = len(piece) + (sep_len if current else 0)
        if current_len + added_len > chunk_size and current:
            chunks.append(separator.join(current).strip())
            # Redémarre avec un overlap textuel du chunk précédent.
            overlap_text = chunks[-1][-chunk_overlap:] if chunk_overlap else ""
            current = [overlap_text, piece] if overlap_text else [piece]
            current_len = sum(len(c) for c in current) + sep_len * (len(current) - 1)
        else:
            current.append(piece)
            current_len += added_len

    if current:
        chunks.append(separator.join(current).strip())
    return [c for c in chunks if c]


def _sliding_window(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Fenêtre glissante brute quand aucun séparateur n'est trouvé."""
    step = chunk_size - chunk_overlap
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]
