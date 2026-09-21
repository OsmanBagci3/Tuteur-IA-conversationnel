# Tuteur IA conversationnel

> Assistant pédagogique conversationnel basé sur un **RAG multimodal** (texte + images) qui enseigne un sujet à partir d'un corpus documentaire, pose des questions à l'apprenant, évalue ses réponses et adapte son niveau (débutant / intermédiaire / avancé).
>
> Sujet de la V1 : **fondamentaux du NLP**, à partir d'un cours PDF de ~95 pages.

**⚠️ Statut : en construction (V1 / MVP).** Ce README sera enrichi au fur et à mesure des phases.

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

1. **Ingère** un corpus (PDF) et le rend interrogeable par recherche sémantique multimodale.
2. **Explique** un concept clairement, en s'appuyant sur des passages sourcés du corpus.
3. **Interroge** l'apprenant pour vérifier sa compréhension.
4. **S'adapte** à son niveau selon la qualité des réponses (3 paliers).
5. **Cite** ses sources (extraits texte et schémas) pour rester fiable et traçable.

## Fonctionnalités V1

- [ ] Ingestion PDF → chunks texte + images extraites
- [ ] Indexation multimodale dans **ChromaDB** (2 collections : texte + image)
- [ ] Retrieval hybride avec fusion **RRF** (Reciprocal Rank Fusion)
- [ ] Auto-diagnostic initial pour placer l'apprenant
- [ ] Boucle pédagogique `explain → quiz → evaluate → adapt`
- [ ] API REST (**FastAPI**)
- [ ] UI de démo (**Streamlit**) avec affichage des sources texte + images
- [ ] Conteneurisation Docker

## Architecture

*Schéma détaillé à venir dans `docs/architecture.md`.*

```
PDF ─► [Ingestion : texte + images]
         │
         ├─► Sentence-Transformers ──► Chroma "text"
         └─► OpenCLIP (image) ───────► Chroma "image"

Question ─► Retriever hybride (RRF) ─► Contexte multimodal ─► Pixtral (Mistral) ─► Réponse + citations
                                                                    ▲
                                         Tutor Engine (état + niveau + objectifs) ─┘
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
| Config | Pydantic Settings + `.env` | Type-safe, secrets isolés |
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

*À compléter au fil des phases. Aperçu :*

```bash
# Ingestion (une fois)
uv run python scripts/ingest.py

# API
uv run uvicorn src.tuteur_ia.api.main:app --reload

# UI (dans un autre terminal)
uv run streamlit run src/tuteur_ia/ui/streamlit_app.py
```

## Structure du dépôt

```
tuteur_IA/
├── config/                     # Settings, prompts, parcours pédagogique
│   ├── settings.py
│   ├── learning_path.yaml
│   └── prompts/
├── src/tuteur_ia/
│   ├── ingestion/              # PDF → texte + images → embeddings → Chroma
│   ├── rag/                    # Retrieval multimodal + fusion RRF
│   ├── llm/                    # Abstraction LLMClient + providers
│   ├── tutor/                  # Machine à états pédagogique + placement
│   ├── api/                    # FastAPI
│   └── ui/                     # Streamlit
├── scripts/
│   └── ingest.py               # CLI d'ingestion
├── data/
│   ├── raw/                    # Corpus source (gitignoré)
│   ├── extracted/images/       # Images extraites du PDF
│   └── chroma/                 # Index vectoriel persistant
├── tests/
├── docs/
├── pyproject.toml
└── .env.example
```

## Limites connues & évolutions

**Hors périmètre V1** (listés comme évolutions futures) :

- Authentification / multi-utilisateurs
- Multi-corpus / multi-sujets
- Fine-tuning d'un modèle
- Persistance avancée de la progression apprenant
- Déploiement cloud
- Gamification poussée
- Évaluation offline systématique (RAGAS, etc.)

## Licence

MIT — voir `LICENSE`.
