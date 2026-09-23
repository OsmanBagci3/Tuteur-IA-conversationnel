"""CLI interactive pour la P3 : session complète de tutorat (placement -> parcours).

Usage :
    uv run python scripts/tutor_cli.py
    uv run python scripts/tutor_cli.py --max-objectives 2 --skip-placement --level beginner
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.settings import settings  # noqa: E402
from tuteur_ia.ingestion.embedders import CLIPImageEmbedder, TextEmbedder  # noqa: E402
from tuteur_ia.llm.providers.mistral import MistralClient  # noqa: E402
from tuteur_ia.rag.retriever import HybridRetriever  # noqa: E402
from tuteur_ia.tutor.engine import TutorEngine  # noqa: E402
from tuteur_ia.tutor.objectives import load_learning_path  # noqa: E402
from tuteur_ia.tutor.state import Level, Phase  # noqa: E402

SEPARATOR = "-" * 70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Session interactive du tuteur IA.")
    parser.add_argument(
        "--max-objectives",
        type=int,
        default=None,
        help="Limite le nombre d'objectifs du parcours (utile pour tester rapidement).",
    )
    parser.add_argument(
        "--skip-placement",
        action="store_true",
        help="Saute l'auto-diagnostic initial et démarre directement à --level.",
    )
    parser.add_argument(
        "--level",
        choices=[level.value for level in Level],
        default=Level.INTERMEDIATE.value,
        help="Niveau de départ si --skip-placement est utilisé (défaut : intermediate).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING,  # on garde la sortie CLI propre, peu de bruit
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    print("Chargement des modèles (peut prendre quelques secondes)...")

    learning_path = load_learning_path(settings.learning_path_file)
    if args.max_objectives is not None:
        learning_path = dataclasses.replace(
            learning_path, objectives=learning_path.objectives[: args.max_objectives]
        )
    text_embedder = TextEmbedder(settings.text_embedding_model)
    clip_embedder = CLIPImageEmbedder(settings.clip_model_name, settings.clip_pretrained)
    retriever = HybridRetriever(
        chroma_dir=settings.chroma_dir,
        text_embedder=text_embedder,
        clip_embedder=clip_embedder,
        rrf_k=settings.rag_rrf_k,
    )
    llm = MistralClient(api_key=settings.mistral_api_key, model=settings.mistral_model)
    engine = TutorEngine(retriever=retriever, llm=llm, learning_path=learning_path)

    print(f"\n=== Tuteur IA — {learning_path.subject} ===")
    print(f"Parcours : {len(learning_path.objectives)} objectifs.\n")

    # --- Auto-diagnostic initial ---
    if args.skip_placement:
        engine.state.level = Level(args.level)
        engine.state.phase = Phase.EXPLAIN
        print(f"\n(Auto-diagnostic sauté — niveau forcé à {engine.state.level.value.upper()})\n")
    else:
        print(SEPARATOR)
        print("AUTO-DIAGNOSTIC INITIAL — quelques questions pour évaluer ton niveau de départ.")
        print(SEPARATOR)
        placement_turns = engine.start_placement()
        for turn in placement_turns:
            print(f"\n[{turn.index + 1}/{turn.total}] ({turn.objective_title})")
            print(turn.question)
            answer = input("Ta réponse > ").strip()
            engine.submit_placement_answer(turn.index, answer)

        level = engine.finalize_placement()
        print(f"\n→ Niveau initial estimé : {level.value.upper()}\n")

    # --- Boucle principale : explain -> quiz -> evaluate -> adapt ---
    while not engine.is_done():
        print(SEPARATOR)
        explain_turn = engine.explain_current_objective()
        print(f"OBJECTIF : {explain_turn.objective_title}  (niveau {engine.state.level.value})")
        print(SEPARATOR)
        print(f"\n{explain_turn.text}\n")
        if explain_turn.source_pages:
            print(f"(Sources : pages {', '.join(map(str, explain_turn.source_pages))})")

        quiz_turn = engine.generate_quiz()
        print(f"\n❓ {quiz_turn.question}")
        answer = input("Ta réponse > ").strip()

        eval_turn = engine.evaluate_answer(answer)
        print(f"\nScore : {eval_turn.score}/5")
        print(f"Feedback : {eval_turn.feedback}")
        if eval_turn.level_changed:
            print(f"→ Niveau ajusté : {eval_turn.level.value.upper()}")
        if not eval_turn.done:
            print(f"\nObjectif suivant : {eval_turn.next_objective_title}")
        print()

    print(SEPARATOR)
    print("Parcours terminé ! Bravo.")
    print(SEPARATOR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
