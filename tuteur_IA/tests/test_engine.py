from __future__ import annotations

import tempfile
from pathlib import Path

from tests.helpers import FakeLLM, FakeRetriever, make_learning_path
from tuteur_ia.observability.tracer import TraceLogger
from tuteur_ia.tutor.engine import TutorEngine
from tuteur_ia.tutor.state import Level, Phase


def test_adapt_level_moves_up_on_high_score():
    assert TutorEngine._adapt_level(Level.INTERMEDIATE, 5) == Level.ADVANCED


def test_adapt_level_moves_down_on_low_score():
    assert TutorEngine._adapt_level(Level.INTERMEDIATE, 1) == Level.BEGINNER


def test_adapt_level_stays_at_ceiling():
    assert TutorEngine._adapt_level(Level.ADVANCED, 5) == Level.ADVANCED


def test_adapt_level_stays_at_floor():
    assert TutorEngine._adapt_level(Level.BEGINNER, 1) == Level.BEGINNER


def test_adapt_level_mid_score_does_not_change_level():
    assert TutorEngine._adapt_level(Level.INTERMEDIATE, 3) == Level.INTERMEDIATE


def _make_engine(scores: list[int]) -> TutorEngine:
    tracer = TraceLogger(Path(tempfile.gettempdir()) / "tuteur_ia_test_traces.jsonl")
    return TutorEngine(
        retriever=FakeRetriever(),  # type: ignore[arg-type]
        llm=FakeLLM(scores=scores),
        learning_path=make_learning_path(3),
        tracer=tracer,
    )


def test_start_placement_generates_one_question_per_probe_objective():
    engine = _make_engine(scores=[3, 3, 3])
    turns = engine.start_placement()
    assert len(turns) == 3
    assert [t.index for t in turns] == [0, 1, 2]
    assert engine.state.phase == Phase.PLACEMENT


def test_finalize_placement_sets_level_and_switches_to_explain_phase():
    engine = _make_engine(scores=[5, 5, 5])
    engine.start_placement()
    for i in range(3):
        engine.submit_placement_answer(i, "une réponse")
    level = engine.finalize_placement()
    assert level == Level.ADVANCED
    assert engine.state.phase == Phase.EXPLAIN
    assert engine.state.objective_index == 0


def test_full_cycle_from_placement_to_done_adapts_level_each_step():
    # 3 scores de placement (moyenne 3 -> intermediate) puis 3 scores d'objectifs.
    engine = _make_engine(scores=[3, 3, 3, 5, 1, 5])
    engine.start_placement()
    for i in range(3):
        engine.submit_placement_answer(i, "réponse")
    assert engine.finalize_placement() == Level.INTERMEDIATE

    expected = [
        (5, Level.ADVANCED, True, False),
        (1, Level.INTERMEDIATE, True, False),
        (5, Level.ADVANCED, True, True),
    ]
    for score, expected_level, expected_changed, expected_done in expected:
        engine.explain_current_objective()
        engine.generate_quiz()
        result = engine.evaluate_answer("réponse de l'apprenant")
        assert result.score == score
        assert result.level == expected_level
        assert result.level_changed is expected_changed
        assert result.done is expected_done

    assert engine.is_done()
    assert len(engine.state.results) == 3
    assert engine.state.objective_index == 3


def test_evaluate_answer_without_pending_quiz_question_raises():
    engine = _make_engine(scores=[])
    try:
        engine.evaluate_answer("réponse")
    except RuntimeError:
        pass
    else:
        raise AssertionError("evaluate_answer() aurait dû lever une RuntimeError")
