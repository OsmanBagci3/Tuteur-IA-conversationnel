from __future__ import annotations

import tempfile
from pathlib import Path

from tests.helpers import FakeLLM, FakeRetriever, make_learning_path
from tuteur_ia.observability.tracer import TraceLogger
from tuteur_ia.tutor.conversation import ChatSession
from tuteur_ia.tutor.engine import TutorEngine


def _make_session(scores: list[int]) -> ChatSession:
    tracer = TraceLogger(Path(tempfile.gettempdir()) / "tuteur_ia_test_traces.jsonl")
    engine = TutorEngine(
        retriever=FakeRetriever(),  # type: ignore[arg-type]
        llm=FakeLLM(scores=scores),
        learning_path=make_learning_path(3),
        tracer=tracer,
    )
    return ChatSession(engine)


def test_start_returns_first_placement_question():
    session = _make_session(scores=[3, 3, 3])
    turns = session.start()
    assert len(turns) == 1
    assert turns[0].kind == "placement_question"


def test_placement_questions_are_asked_one_by_one():
    session = _make_session(scores=[3, 3, 3])
    session.start()
    turns = session.send("réponse 1")
    assert len(turns) == 1
    assert turns[0].kind == "placement_question"


def test_end_of_placement_emits_level_and_start_transition_messages():
    session = _make_session(scores=[3, 3, 3])
    session.start()
    session.send("réponse 1")
    session.send("réponse 2")
    turns = session.send("réponse 3")  # dernière réponse de placement (3 probes)
    kinds = [t.kind for t in turns]
    texts = [t.text for t in turns]
    assert kinds == ["system", "system", "explain", "quiz"]
    assert texts[0].startswith("Niveau initial estimé")
    assert texts[1].startswith("🚀 On démarre avec : Objectif")


def test_quiz_answer_before_last_objective_emits_next_objective_transition():
    # 3 scores de placement + 1 score pour le premier objectif du parcours principal.
    session = _make_session(scores=[3, 3, 3, 3])
    session.start()
    session.send("réponse 1")
    session.send("réponse 2")
    session.send("réponse 3")  # termine le placement, démarre le premier objectif

    turns = session.send("réponse au quiz")
    kinds = [t.kind for t in turns]
    texts = [t.text for t in turns]
    assert kinds == ["evaluation", "system", "explain", "quiz"]
    assert texts[1].startswith("➡️ Objectif suivant : Objectif")


def test_last_objective_emits_completion_message_instead_of_transition():
    # 3 objectifs au total : 3 scores de placement + 3 scores pour clore le parcours.
    session = _make_session(scores=[3, 3, 3, 3, 3, 3])
    session.start()
    session.send("réponse 1")
    session.send("réponse 2")
    session.send("réponse 3")  # démarre l'objectif 0

    session.send("quiz 0")  # clôt l'objectif 0, démarre l'objectif 1
    session.send("quiz 1")  # clôt l'objectif 1, démarre l'objectif 2
    turns = session.send("quiz 2")  # clôt l'objectif 2 (dernier) -> parcours terminé

    kinds = [t.kind for t in turns]
    texts = [t.text for t in turns]
    assert kinds == ["evaluation", "system"]
    assert texts[1] == "Parcours terminé, bravo !"


def test_sending_after_completion_returns_a_closing_message():
    session = _make_session(scores=[3, 3, 3, 3, 3, 3])
    session.start()
    session.send("réponse 1")
    session.send("réponse 2")
    session.send("réponse 3")
    session.send("quiz 0")
    session.send("quiz 1")
    session.send("quiz 2")

    turns = session.send("encore un message")
    assert len(turns) == 1
    assert "terminé" in turns[0].text
