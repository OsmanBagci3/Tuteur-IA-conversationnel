from __future__ import annotations

from pathlib import Path

from tuteur_ia.llm.base import LLMClient, Message
from tuteur_ia.tutor.evaluator import evaluate_answer


class _ScriptedLLM(LLMClient):
    model = "fake"

    def __init__(self, response: str) -> None:
        self._response = response
        self.last_prompt: str | None = None

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        self.last_prompt = messages[0].content
        return self._response


def _make_prompt(tmp_path: Path) -> Path:
    path = tmp_path / "evaluate.txt"
    path.write_text("{objective_title}|{level}|{question}|{answer}|{context}", encoding="utf-8")
    return path


def test_parses_well_formed_score_and_feedback(tmp_path: Path):
    llm = _ScriptedLLM("SCORE: 4\nFEEDBACK: Bonne réponse, bien expliquée.")
    result = evaluate_answer(
        llm,
        _make_prompt(tmp_path),
        "Tokenisation",
        "intermediate",
        "Q ?",
        "Ma réponse.",
        "Contexte.",
    )
    assert result.score == 4
    assert result.feedback == "Bonne réponse, bien expliquée."


def test_parsing_is_case_insensitive(tmp_path: Path):
    llm = _ScriptedLLM("score: 2\nfeedback: Incomplet.")
    result = evaluate_answer(llm, _make_prompt(tmp_path), "T", "beginner", "Q", "A", "C")
    assert result.score == 2
    assert result.feedback == "Incomplet."


def test_malformed_response_falls_back_to_score_three(tmp_path: Path):
    llm = _ScriptedLLM("Je ne sais pas trop quoi répondre ici.")
    result = evaluate_answer(llm, _make_prompt(tmp_path), "T", "beginner", "Q", "A", "C")
    assert result.score == 3
    assert result.feedback == "Je ne sais pas trop quoi répondre ici."


def test_score_outside_1_5_range_is_not_matched_and_falls_back(tmp_path: Path):
    llm = _ScriptedLLM("SCORE: 7\nFEEDBACK: Score hors plage.")
    result = evaluate_answer(llm, _make_prompt(tmp_path), "T", "beginner", "Q", "A", "C")
    assert result.score == 3


def test_prompt_template_is_filled_with_all_fields(tmp_path: Path):
    llm = _ScriptedLLM("SCORE: 1\nFEEDBACK: x")
    evaluate_answer(
        llm, _make_prompt(tmp_path), "Titre", "advanced", "Question ?", "Réponse", "Ctx"
    )
    assert llm.last_prompt == "Titre|advanced|Question ?|Réponse|Ctx"
