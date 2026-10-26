"""FAHR-Face Live: a lightweight local webcam demo on top of the FAHR-Face models.

This package deliberately keeps its top-level import light (no torch, cv2 or
transformers) so that pure-numpy parts such as :mod:`app.align` can be imported
in minimal environments.  It inserts the repository root at the front of
``sys.path`` so the repository's ``ViTForAgeEstimation.py`` and
``ViTForSurvivalAnalysis.py`` can be imported unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

__version__ = "0.1.0"

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

__all__ = ["__version__", "REPO_ROOT"]
