"""UI Streamlit du Tuteur IA — interface de chat pédagogique.

Lancement :
    uv run streamlit run src/tuteur_ia/ui/app.py --server.port 8501
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings  # noqa: E402

API_BASE_URL = settings.api_base_url
TIMEOUT = 120  # les appels LLM peuvent prendre plusieurs secondes


def _post(path: str, json: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        response = httpx.post(f"{API_BASE_URL}{path}", json=json, timeout=TIMEOUT)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        st.error(f"Impossible de joindre l'API ({API_BASE_URL}) : {exc}")
        st.stop()
    return response.json()


def _apply_session_out(data: dict[str, Any]) -> None:
    st.session_state.session_id = data["session_id"]
    st.session_state.phase = data["phase"]
    st.session_state.level = data["level"]
    st.session_state.objective_index = data["objective_index"]
    st.session_state.total_objectives = data["total_objectives"]
    st.session_state.done = data["done"]
    for turn in data["turns"]:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": turn["text"],
                "score": turn.get("score"),
                "source_pages": turn.get("source_pages", []),
                "image_urls": turn.get("image_urls", []),
            }
        )


def _start_session() -> None:
    _apply_session_out(_post("/session"))


def _reset_session() -> None:
    st.session_state.messages = []
    _apply_session_out(_post("/reset", json={"session_id": st.session_state.session_id}))


def _render_message(msg: dict[str, Any]) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        if msg.get("score") is not None:
            st.caption(f"Score : {msg['score']}/5")
        if msg.get("source_pages"):
            pages = ", ".join(str(p) for p in msg["source_pages"])
            st.caption(f"📖 Sources : pages {pages}")
        for url in msg.get("image_urls", []):
            st.image(f"{API_BASE_URL}{url}")


st.set_page_config(page_title="Tuteur IA — NLP", page_icon="🎓")

if "session_id" not in st.session_state:
    st.session_state.messages = []
    st.session_state.session_id = None
    st.session_state.done = False
    with st.spinner("Démarrage de la session..."):
        _start_session()

with st.sidebar:
    st.title("🎓 Tuteur IA")
    st.caption("Fondamentaux du NLP")
    st.metric("Niveau actuel", st.session_state.level.capitalize())
    st.caption(f"Phase : {st.session_state.phase}")
    total = st.session_state.total_objectives
    if total:
        done_count = min(st.session_state.objective_index, total)
        st.progress(done_count / total)
        st.caption(f"Objectif {min(done_count + 1, total)}/{total}")
    if st.button("🔄 Recommencer"):
        _reset_session()
        st.rerun()

st.header("Session de tutorat")

for message in st.session_state.messages:
    _render_message(message)

if st.session_state.done:
    st.success("Parcours terminé, bravo ! Clique sur « Recommencer » pour une nouvelle session.")
else:
    user_input = st.chat_input("Ta réponse...")
    if user_input:
        st.session_state.messages.append(
            {"role": "user", "text": user_input, "score": None, "source_pages": [], "image_urls": []}
        )
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.spinner("Le tuteur réfléchit..."):
            _apply_session_out(
                _post("/message", json={"session_id": st.session_state.session_id, "content": user_input})
            )
        st.rerun()
