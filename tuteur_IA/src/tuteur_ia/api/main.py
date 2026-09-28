"""API FastAPI du Tuteur IA.

Lancement :
    uv run uvicorn tuteur_ia.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

# `config` est un package racine (hors `src/`) : on l'ajoute au path avant de
# l'importer, quelle que soit la façon dont uvicorn a été invoqué.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from config.settings import settings  # noqa: E402
from tuteur_ia.api.dependencies import Resources, build_resources  # noqa: E402
from tuteur_ia.api.schemas import ChatTurnOut, MessageIn, ResetIn, SessionOut  # noqa: E402
from tuteur_ia.tutor.conversation import ChatSession, ChatTurn  # noqa: E402
from tuteur_ia.tutor.engine import TutorEngine  # noqa: E402

SESSIONS: dict[str, ChatSession] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.resources = build_resources()
    yield


app = FastAPI(title="Tuteur IA", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://localhost:{settings.ui_port}", f"http://127.0.0.1:{settings.ui_port}"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sert les images extraites du PDF pour que l'UI puisse les afficher par URL.
app.mount("/images", StaticFiles(directory=settings.images_dir), name="images")


def _new_engine() -> TutorEngine:
    resources: Resources = app.state.resources
    return TutorEngine(
        retriever=resources.retriever,
        llm=resources.llm,
        learning_path=resources.learning_path,
    )


def _get_session(session_id: str) -> ChatSession:
    chat = SESSIONS.get(session_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Session inconnue.")
    return chat


def _image_url(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(settings.images_dir)
    except ValueError:
        return str(path)
    return f"/images/{rel.as_posix()}"


def _to_turn_out(turn: ChatTurn) -> ChatTurnOut:
    return ChatTurnOut(
        kind=turn.kind,
        text=turn.text,
        source_pages=turn.source_pages,
        image_urls=[_image_url(p) for p in turn.image_paths],
        score=turn.score,
    )


def _to_session_out(session_id: str, chat: ChatSession, turns: list[ChatTurn]) -> SessionOut:
    state = chat.engine.state
    return SessionOut(
        session_id=session_id,
        phase=state.phase.value,
        level=state.level.value,
        objective_index=state.objective_index,
        total_objectives=chat.engine.total_objectives,
        done=chat.engine.is_done(),
        turns=[_to_turn_out(t) for t in turns],
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/session", response_model=SessionOut)
def create_session() -> SessionOut:
    chat = ChatSession(_new_engine())
    turns = chat.start()
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = chat
    return _to_session_out(session_id, chat, turns)


@app.post("/message", response_model=SessionOut)
def send_message(payload: MessageIn) -> SessionOut:
    chat = _get_session(payload.session_id)
    turns = chat.send(payload.content)
    return _to_session_out(payload.session_id, chat, turns)


@app.post("/reset", response_model=SessionOut)
def reset_session(payload: ResetIn) -> SessionOut:
    chat = ChatSession(_new_engine())
    turns = chat.start()
    SESSIONS[payload.session_id] = chat
    return _to_session_out(payload.session_id, chat, turns)
