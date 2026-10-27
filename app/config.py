"""Runtime settings, read from ``FAHR_*`` environment variables (spec section 4.1)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Mapping, Optional

DEFAULT_MODEL_CACHE_DIR = "~/.cache/fahr-face"

_TRUE = {"1", "true", "yes", "on", "y", "t"}
_FALSE = {"0", "false", "no", "off", "n", "f", ""}


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"{name}={raw!r} is not a boolean (use 1/0, true/false)")


def _env_int(env: Mapping[str, str], name: str, default: int, *, minimum: Optional[int] = None) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not an integer") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name}={raw!r} must be >= {minimum}")
    return value


def _env_float(env: Mapping[str, str], name: str, default: Optional[float]) -> Optional[float]:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a number") from exc


def _env_str(env: Mapping[str, str], name: str, default: str) -> str:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def _env_opt_str(env: Mapping[str, str], name: str) -> Optional[str]:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip()


@dataclass(frozen=True)
class Settings:
    """All knobs of the demo server.  See :meth:`from_env` for the variables."""

    age_checkpoint: str = "checkpoints/fahr_faceage.pth"
    survival_checkpoint: str = "checkpoints/fahr_facesurvival.pth"
    detector: str = "auto"  # auto | yunet | retinaface
    detection_threshold: Optional[float] = None  # None -> detector default
    device: str = "auto"  # auto | cpu | cuda | mps
    allow_random_weights: bool = False
    max_faces: int = 5
    max_upload_bytes: int = 8 * 1024 * 1024
    model_cache_dir: str = DEFAULT_MODEL_CACHE_DIR
    yunet_model: Optional[str] = None
    host: str = "127.0.0.1"
    port: int = 8000
    # Where the frontend files live (default: app/static next to the package).
    static_dir: Optional[str] = field(default=None)

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "Settings":
        """Build settings from ``FAHR_*`` variables in ``environ`` (default ``os.environ``).

        Raises:
            ValueError: for values that cannot be parsed or are out of range.
        """
        env: Mapping[str, str] = os.environ if environ is None else environ
        return cls(
            age_checkpoint=_env_str(env, "FAHR_AGE_CHECKPOINT", cls.age_checkpoint),
            survival_checkpoint=_env_str(env, "FAHR_SURVIVAL_CHECKPOINT", cls.survival_checkpoint),
            detector=_env_str(env, "FAHR_DETECTOR", "auto").lower(),
            detection_threshold=_env_float(env, "FAHR_DETECTION_THRESHOLD", None),
            device=_env_str(env, "FAHR_DEVICE", "auto").lower(),
            allow_random_weights=_env_bool(env, "FAHR_ALLOW_RANDOM_WEIGHTS", False),
            max_faces=_env_int(env, "FAHR_MAX_FACES", 5, minimum=1),
            max_upload_bytes=_env_int(env, "FAHR_MAX_UPLOAD_BYTES", 8 * 1024 * 1024, minimum=1024),
            model_cache_dir=_env_str(env, "FAHR_MODEL_CACHE_DIR", DEFAULT_MODEL_CACHE_DIR),
            yunet_model=_env_opt_str(env, "FAHR_YUNET_MODEL"),
            host=_env_str(env, "FAHR_HOST", "127.0.0.1"),
            port=_env_int(env, "FAHR_PORT", 8000, minimum=1),
            static_dir=_env_opt_str(env, "FAHR_STATIC_DIR"),
        )

    @property
    def model_cache_path(self) -> str:
        return os.path.expanduser(self.model_cache_dir)

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


__all__ = ["Settings", "DEFAULT_MODEL_CACHE_DIR"]
