from __future__ import annotations

import json
from pathlib import Path

from tuteur_ia.observability.tracer import TraceLogger


def test_log_appends_one_json_line_with_event_and_fields(tmp_path: Path):
    log_path = tmp_path / "logs" / "traces.jsonl"
    tracer = TraceLogger(log_path)
    tracer.log("retrieval", objective_id="obj1", num_texts=3)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "retrieval"
    assert record["objective_id"] == "obj1"
    assert record["num_texts"] == 3
    assert "timestamp" in record


def test_log_creates_parent_directories(tmp_path: Path):
    log_path = tmp_path / "a" / "b" / "traces.jsonl"
    TraceLogger(log_path).log("event")
    assert log_path.exists()


def test_multiple_logs_append_multiple_lines(tmp_path: Path):
    tracer = TraceLogger(tmp_path / "traces.jsonl")
    tracer.log("a")
    tracer.log("b")
    lines = (tmp_path / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_timed_context_manager_logs_latency_ms(tmp_path: Path):
    log_path = tmp_path / "traces.jsonl"
    tracer = TraceLogger(log_path)
    with tracer.timed("explain", objective_id="obj1"):
        pass
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["event"] == "explain"
    assert record["objective_id"] == "obj1"
    assert isinstance(record["latency_ms"], (int, float))


def test_log_never_raises_when_path_is_unwritable(tmp_path: Path):
    # Un fichier existant à la place du dossier parent rend l'écriture impossible ;
    # `log()` doit l'avaler silencieusement (observabilité = bonus, jamais bloquant).
    blocking_file = tmp_path / "not_a_dir"
    blocking_file.write_text("x", encoding="utf-8")
    tracer = TraceLogger(blocking_file / "traces.jsonl")
    tracer.log("event")  # ne doit pas lever
