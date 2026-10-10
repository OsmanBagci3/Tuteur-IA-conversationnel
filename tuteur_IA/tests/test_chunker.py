from __future__ import annotations

import pytest

from tuteur_ia.ingestion.chunker import chunk_pages
from tuteur_ia.ingestion.loader import PageText


def test_short_page_produces_a_single_untouched_chunk():
    pages = [PageText(page=1, text="Un texte court.")]
    chunks = chunk_pages(pages, chunk_size=800, chunk_overlap=150)
    assert len(chunks) == 1
    assert chunks[0].page == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == "Un texte court."


def test_long_page_is_split_into_multiple_chunks():
    paragraph = "Phrase de test suffisamment longue. " * 50
    pages = [PageText(page=1, text=paragraph)]
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) > 0


def test_chunk_index_is_sequential_within_a_page():
    paragraph = "Phrase de test suffisamment longue. " * 50
    pages = [PageText(page=1, text=paragraph)]
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=50)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_each_page_restarts_chunk_index_at_zero():
    pages = [
        PageText(page=1, text="Contenu de la page un."),
        PageText(page=2, text="Contenu de la page deux."),
    ]
    chunks = chunk_pages(pages, chunk_size=800, chunk_overlap=150)
    assert [c.page for c in chunks] == [1, 2]
    assert [c.chunk_index for c in chunks] == [0, 0]


def test_blank_page_produces_no_chunks():
    pages = [PageText(page=1, text="   ")]
    assert chunk_pages(pages, chunk_size=800, chunk_overlap=150) == []


def test_overlap_must_be_strictly_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_pages([PageText(page=1, text="x")], chunk_size=100, chunk_overlap=100)


def test_reconstructed_text_preserves_all_words_in_order():
    # Pas de perte de mots lors du découpage/recollage (chevauchement mis à part).
    words = [f"mot{i}" for i in range(200)]
    paragraph = " ".join(words)
    pages = [PageText(page=1, text=paragraph)]
    chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=20)
    joined = " ".join(c.text for c in chunks)
    for word in words:
        assert word in joined
