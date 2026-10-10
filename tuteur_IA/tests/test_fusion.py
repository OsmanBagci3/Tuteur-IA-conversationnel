from __future__ import annotations

from tuteur_ia.rag.fusion import reciprocal_rank_fusion


def test_single_list_preserves_relative_order():
    result = reciprocal_rank_fusion([["a", "b", "c"]])
    assert [item_id for item_id, _ in result] == ["a", "b", "c"]


def test_item_present_in_multiple_lists_outranks_single_list_item():
    result = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    ranking = [item_id for item_id, _ in result]
    # "a" et "b" apparaissent en tête des deux listes : ils doivent dominer "c"/"d",
    # présents dans une seule liste.
    assert ranking.index("a") < ranking.index("c")
    assert ranking.index("b") < ranking.index("d")


def test_score_matches_rrf_formula():
    result = dict(reciprocal_rank_fusion([["a", "b"]], k=60))
    assert result["a"] == 1 / 61
    assert result["b"] == 1 / 62


def test_scores_from_multiple_lists_are_summed():
    result = dict(reciprocal_rank_fusion([["a"], ["a"]], k=60))
    assert result["a"] == 2 * (1 / 61)


def test_empty_input_returns_empty_ranking():
    assert reciprocal_rank_fusion([]) == []


def test_empty_sublists_are_ignored():
    assert reciprocal_rank_fusion([[], []]) == []
