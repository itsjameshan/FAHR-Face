"""Shared pytest fixtures.

Heavy imports (torch, cv2, fastapi) are done lazily inside fixtures so that
``tests/test_align.py`` can run in a minimal environment without them.
"""

from __future__ import annotations

import os

import pytest

# Keep TensorFlow quiet if the optional retina-face fidelity test runs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


@pytest.fixture(scope="session")
def random_settings():
    """Settings that never find a checkpoint and allow random age weights."""
    from app.config import Settings

    return Settings(
        allow_random_weights=True,
        age_checkpoint="/nonexistent/age.pth",
        survival_checkpoint="/nonexistent/surv.pth",
        max_upload_bytes=1_000_000,
        max_faces=5,
    )


@pytest.fixture(scope="session")
def face_models(random_settings):
    from app.models import FaceModels

    return FaceModels(random_settings)


@pytest.fixture(scope="session")
def client(random_settings):
    """A TestClient whose lifespan has run (detector + models loaded)."""
    from fastapi.testclient import TestClient

    from app.server import create_app

    with TestClient(create_app(random_settings)) as c:
        yield c
