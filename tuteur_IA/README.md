# Tuteur IA conversationnel

> Assistant pédagogique conversationnel basé sur un **RAG multimodal** (texte + images) qui enseigne un sujet à partir d'un corpus documentaire, pose des questions à l'apprenant, évalue ses réponses et adapte son niveau (débutant / intermédiaire / avancé).
>
> Sujet de la V1 : **fondamentaux du NLP**, à partir d'un cours PDF de ~95 pages.


![Démo](docs/peek_3.gif)

---

## Table des matières

- [Objectif](#objectif)
- [Fonctionnalités V1](#fonctionnalités-v1)
- [Architecture](#architecture)
- [Stack technique](#stack-technique)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du dépôt](#structure-du-dépôt)
- [Limites connues & évolutions](#limites-connues--évolutions)

---

## Objectif

Construire un tuteur conversationnel qui :

1. Ingère un corpus (PDF) et le rend interrogeable par recherche sémantique multimodale.
2. Explique un concept clairement, en s'appuyant sur des passages sourcés du corpus.
3. Interroge l'apprenant pour vérifier sa compréhension.
4. S'adapte à son niveau selon la qualité des réponses (3 paliers).
5. Cite ses sources (extraits texte et schémas) pour rester fiable et traçable.

## Fonctionnalités V1

- Ingestion PDF → chunks texte + images extraites
- Indexation multimodale dans **ChromaDB** (2 collections : texte + image)
- Retrieval hybride avec fusion **RRF** (Reciprocal Rank Fusion)
- Garde-fou anti-hallucination (seuil de score + prompt renforcé)
- Auto-diagnostic initial pour placer l'apprenant
- Boucle pédagogique `explain → quiz → evaluate → adapt`
- API REST (**FastAPI**) : `/session`, `/message`, `/reset`
- UI de démo (**Streamlit**) avec chat, badge niveau, sources texte + images
- Conteneurisation Docker (API + UI, `docker-compose`)
- Reranking cross-encoder (`sentence-transformers.CrossEncoder`) sur les résultats du retriever
- Traçabilité : logs JSONL locaux (latence, scores, événements du moteur pédagogique)
- Tests automatisés (pytest, 56 tests) et CI (GitHub Actions)

## Architecture

```
PDF ─► [Ingestion : texte + images]
         │
         ├─► Sentence-Transformers ──► Chroma "text"
         └─► OpenCLIP (image) ───────► Chroma "image"

┌─────────────────────────── Session de tutorat ───────────────────────────┐
│                                                                          │
│  UI Streamlit ──HTTP──► API FastAPI                                     │
│                              │                                          │
│                              ▼                                          │
│                        ChatSession (séquenceur de tours)                │
│                              │                                          │
│                              ▼                                          │
│                        TutorEngine (état : niveau, phase, objectif)     │
│                              │                                          │
│              ┌───────────────┼───────────────┐                          │
│              ▼               ▼               ▼                         │
│      HybridRetriever   Garde-fou score   MistralClient (Pixtral)        │
│      (RRF texte+image)  (RAG_MIN_SCORE)   (texte + images en entrée)    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

## Stack technique

| Couche | Choix | Justification |
|---|---|---|
| Langage | Python ≥ 3.11 | Standard |
| LLM | Mistral `pixtral-12b` (VLM) | Free tier, gère images en entrée, cohérence provider |
| Embeddings texte | `sentence-transformers` (MiniLM) | Léger, universel, offline |
| Embeddings image | `open-clip-torch` (ViT-B-32) | CLIP moderne, checkpoints ouverts |
| Base vectorielle | ChromaDB (mode persistant) | Zéro infra, deux collections |
| API | FastAPI + Uvicorn | Standard Python moderne, async |
| UI | Streamlit | Démo rapide, affichage images natif |
| Packaging | `uv` + `hatchling` (src layout) | Rapide, standard PEP 621 / 735 |

## Installation

Prérequis : Python ≥ 3.11 et [`uv`](https://docs.astral.sh/uv/) installés.

```bash
# 1. Cloner
git clone <url-du-repo>
cd Tuteur-IA-conversationnel/tuteur_IA

# 2. Installer les dépendances
uv sync

# 3. Configurer les secrets
cp .env.example .env
# → éditer .env et renseigner MISTRAL_API_KEY

# 4. Placer le corpus dans data/raw/ (voir CORPUS_PATH dans .env)
```

## Utilisation

```bash
# 1. Ingestion du corpus (une fois, ou après modification du PDF)
uv run python scripts/ingest.py --reset

# 2. Test RAG + LLM en ligne de commande (sans le moteur pédagogique)
uv run python scripts/query.py "Qu'est-ce que le NLP ?"

# 3. Session de tutorat complète en CLI (placement + parcours)
uv run python scripts/tutor_cli.py
# Options utiles pour un test rapide :
uv run python scripts/tutor_cli.py --max-objectives 2 --skip-placement --level beginner

# 4. API + UI (deux terminaux)
uv run uvicorn tuteur_ia.api.main:app --reload --port 8001
uv run streamlit run src/tuteur_ia/ui/app.py --server.port 8501
# → ouvrir http://localhost:8501
```

### Avec Docker

```bash
# Prérequis : avoir déjà lancé l'ingestion en local (data/chroma et
# data/extracted/images sont montés en volumes, pas rebuild dans l'image)
uv run python scripts/ingest.py --reset

docker compose up --build
# → API sur http://localhost:8001, UI sur http://localhost:8501
```

## Structure du dépôt

```
tuteur_IA/
├── config/                     # Settings, prompts, parcours
│   ├── settings.py
│   ├── learning_path.yaml
│   └── prompts/                
├── src/tuteur_ia/
│   ├── ingestion/              # PDF → texte + images → embeddings → Chroma
│   ├── rag/                    # Retrieval multimodal, fusion RRF, garde-fou, contexte
│   ├── llm/                    # Abstraction LLMClient + providers (Mistral/Pixtral)
│   ├── tutor/                  # État, placement, quiz, évaluation, moteur, conversation
│   ├── api/                    # FastAPI (routes, schémas, ressources partagées)
│   └── ui/                     # Streamlit (chat)
├── scripts/
│   ├── ingest.py                # CLI d'ingestion
│   ├── query.py                 # CLI de test RAG + LLM
│   └── tutor_cli.py             # CLI de session de tutorat complète
├── data/
│   ├── raw/                    # Corpus source (ignoré)
│   ├── extracted/images/       # Images extraites du PDF (ignoré)
│   └── chroma/                 # Index vectoriel persistant (ignoré)
├── tests/
├── docs/
├── Dockerfile.api / Dockerfile.ui / docker-compose.yml
├── pyproject.toml
└── .env.example
```

## Limites connues & évolutions

**Limitations assumées de la V1 (à documenter, pas à cacher) :**

- **Seuil anti-hallucination (`RAG_MIN_SCORE=0.25`)** calibré empiriquement sur seulement
  5 questions de test — un jeu d'évaluation étiqueté plus large permettrait de le fiabiliser.
- **Sessions en mémoire** côté API (pas de base de données) : redémarrer l'API perd toutes
  les sessions en cours. Suffisant pour une démo, pas pour un usage multi-utilisateurs réel.
- **Score image (CLIP) non utilisé pour le garde-fou** : empiriquement peu discriminant
  entre questions sur/hors sujet (voir `JOURNAL.md` pour le détail de l'analyse).
- **Image Docker de l'UI non optimisée** : partage le même environnement complet que l'API
  (torch, chromadb...) alors qu'elle n'a besoin que de `streamlit`/`httpx` — simplification
  assumée (un seul `pyproject.toml`) plutôt que des groupes de dépendances séparés.

**Hors périmètre V1** (évolutions futures) :

- Authentification / multi-utilisateurs
- Multi-corpus / multi-sujets
- Fine-tuning d'un modèle
- Persistance avancée de la progression apprenant (base de données)
- Déploiement cloud
- Gamification poussée
- Évaluation offline systématique (RAGAS, etc.)

## Licence

MIT — voir [`LICENSE`](../LICENSE).
