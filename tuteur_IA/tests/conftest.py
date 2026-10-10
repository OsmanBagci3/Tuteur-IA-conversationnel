"""Bootstrap des imports pour les tests, identique au pattern utilisé par `scripts/*.py` :
le projet n'est pas packagé de façon standard (`config/` vit à la racine, hors `src/`), donc
on ajoute explicitement la racine et `src/` à `sys.path` avant toute collecte de test.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
