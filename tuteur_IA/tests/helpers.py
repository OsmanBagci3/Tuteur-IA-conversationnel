"""Doublures de test (fakes) réutilisées par plusieurs modules de test : un LLM et un
retriever factices qui respectent les interfaces attendues par `TutorEngine`, sans
dépendance réseau, Chroma ou PyTorch.
"""

from __future__ import annotations

from tuteur_ia.llm.base import LLMClient, Message
from tuteur_ia.rag.retriever import RetrievalResult, RetrievedText
from tuteur_ia.tutor.objectives import LearningPath, Objective

# Marqueur unique du template `config/prompts/evaluate.txt`, absent de `quiz.txt`/`explain.txt` :
# sert à distinguer un appel d'évaluation (qui exige un format `SCORE:`/`FEEDBACK:`) d'un appel
# de génération de question/explication (pour lesquels aucun format n'est imposé).
_EVALUATE_MARKER = "RÉPONSE DE L'APPRENANT"


class FakeLLM(LLMClient):
    """LLM factice : rejoue une liste de scores scriptés pour les appels d'évaluation
    (placement et quiz), dans l'ordre où ils sont effectués ; répond un texte générique
    pour les autres appels (génération d'explication / de question de quiz).
    """

    model = "fake-llm"

    def __init__(self, scores: list[int] | None = None) -> None:
        self._scores = iter(scores or [])
        self.calls: list[str] = []

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        prompt = messages[-1].content
        self.calls.append(prompt)
        if _EVALUATE_MARKER in prompt:
            score = next(self._scores)
            return f"SCORE: {score}\nFEEDBACK: Feedback factice pour le score {score}."
        return "Réponse générique du LLM factice."


class FakeRetriever:
    """Retriever factice : renvoie toujours le même extrait texte, au-dessus du seuil
    par défaut du garde-fou anti-hallucination, sans image (évite de manipuler des
    fichiers réels/Chroma dans les tests unitaires).
    """

    def retrieve(self, query: str, k_text: int = 5, k_image: int = 3) -> RetrievalResult:
        text = RetrievedText(
            id="text-1",
            text="Extrait de cours factice pour les tests.",
            page=1,
            chunk_index=0,
            source="test",
            score=0.9,
        )
        return RetrievalResult(texts=[text], images=[], unified_ranking=[("text-1", 1.0)])


def make_learning_path(n: int = 3) -> LearningPath:
    """Construit un parcours pédagogique minimal à `n` objectifs, pour isoler les tests
    du contenu réel de `config/learning_path.yaml`."""
    objectives = [
        Objective(
            id=f"obj{i}",
            title=f"Objectif {i}",
            query_hint=f"objectif {i}",
            summary=f"Résumé de l'objectif {i}.",
        )
        for i in range(n)
    ]
    return LearningPath(
        subject="Test", levels=["beginner", "intermediate", "advanced"], objectives=objectives
    )
