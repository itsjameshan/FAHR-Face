"""Test fixture photos, downloaded on demand from public GitHub repositories.

The photos are cached in ``tests/fixtures/.cache/`` (not part of the repository).
When a photo is not cached and cannot be downloaded, tests that need it skip.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / ".cache"

_BASE = "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/"

FIXTURE_URLS = {
    "obama.jpg": _BASE + "obama.jpg",  # one face
    "two_people.jpg": _BASE + "two_people.jpg",  # two faces
    "biden.jpg": _BASE + "biden.jpg",  # one face
}

DOWNLOAD_TIMEOUT_S = 30


def fixture_path(name: str) -> Path | None:
    """Return the local path of fixture ``name``, downloading it if necessary.

    Returns ``None`` when the file is not cached and the download fails (offline).
    """
    if name not in FIXTURE_URLS:
        raise KeyError(f"unknown fixture {name!r}; known: {sorted(FIXTURE_URLS)}")
    path = CACHE_DIR / name
    if path.is_file() and path.stat().st_size > 0:
        return path
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(FIXTURE_URLS[name], timeout=DOWNLOAD_TIMEOUT_S) as resp:
            data = resp.read()
        if not data:
            return None
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except (urllib.error.URLError, OSError, ValueError):
        try:
            tmp.unlink()
        except OSError:
            pass
        return None
    return path


def require_fixture(name: str) -> Path:
    """Return the fixture path or ``pytest.skip`` when it is unavailable."""
    import pytest

    path = fixture_path(name)
    if path is None:
        pytest.skip(f"fixture {name} not cached and could not be downloaded")
    return path


__all__ = ["CACHE_DIR", "FIXTURE_URLS", "fixture_path", "require_fixture"]
