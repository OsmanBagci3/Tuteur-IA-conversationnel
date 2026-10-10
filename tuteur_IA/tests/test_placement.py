from __future__ import annotations

from tuteur_ia.tutor.objectives import Objective
from tuteur_ia.tutor.placement import initial_level_from_scores, select_placement_objectives
from tuteur_ia.tutor.state import Level


def _objectives(n: int) -> list[Objective]:
    return [
        Objective(id=f"o{i}", title=f"Objectif {i}", query_hint=f"q{i}", summary="")
        for i in range(n)
    ]


def test_returns_all_objectives_when_fewer_than_count():
    objectives = _objectives(2)
    assert select_placement_objectives(objectives, count=3) == objectives


def test_returns_exactly_count_objectives_spanning_start_and_end():
    objectives = _objectives(7)
    probes = select_placement_objectives(objectives, count=3)
    assert len(probes) == 3
    assert probes[0] == objectives[0]
    assert probes[-1] == objectives[-1]


def test_probe_indices_are_deduplicated_and_ordered():
    objectives = _objectives(4)
    probes = select_placement_objectives(objectives, count=3)
    # Pas de doublon d'objectif et ordre croissant préservé.
    assert len(probes) == len({o.id for o in probes})
    ids = [objectives.index(p) for p in probes]
    assert ids == sorted(ids)


def test_initial_level_empty_scores_defaults_to_intermediate():
    assert initial_level_from_scores([]) == Level.INTERMEDIATE


def test_initial_level_low_average_is_beginner():
    assert initial_level_from_scores([1, 2, 2]) == Level.BEGINNER


def test_initial_level_high_average_is_advanced():
    assert initial_level_from_scores([4, 5, 4]) == Level.ADVANCED


def test_initial_level_mid_average_is_intermediate():
    assert initial_level_from_scores([3, 3, 3]) == Level.INTERMEDIATE


def test_initial_level_boundary_average_of_two_is_beginner():
    assert initial_level_from_scores([2, 2]) == Level.BEGINNER


def test_initial_level_boundary_average_of_four_is_advanced():
    assert initial_level_from_scores([4, 4]) == Level.ADVANCED
