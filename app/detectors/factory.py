"""Pick a detector by name."""

from __future__ import annotations

from typing import Optional

from app.detectors.base import FaceDetector

DETECTOR_NAMES = ("auto", "yunet", "retinaface")


def create_detector(
    name: str = "auto",
    threshold: Optional[float] = None,
    *,
    model_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> FaceDetector:
    """Create a detector.

    Args:
        name: ``"auto"`` (= YuNet), ``"yunet"`` or ``"retinaface"``.
        threshold: score threshold; ``None`` uses the detector's default.
        model_path: explicit YuNet ONNX path (``FAHR_YUNET_MODEL``).
        cache_dir: YuNet model cache directory (``FAHR_MODEL_CACHE_DIR``).

    Raises:
        ValueError: unknown detector name.
        ImportError: ``retinaface`` requested but the package is not installed.
        app.detectors.yunet.YuNetModelUnavailable: YuNet model cannot be obtained.
    """
    key = (name or "auto").strip().lower()
    if key in ("auto", "yunet"):
        from app.detectors.yunet import YuNetDetector

        return YuNetDetector(model_path=model_path, threshold=threshold, cache_dir=cache_dir)
    if key == "retinaface":
        from app.detectors.retinaface import RetinaFaceDetector

        return RetinaFaceDetector(threshold=threshold)
    raise ValueError(f"Unknown detector {name!r}; expected one of {', '.join(DETECTOR_NAMES)}")


__all__ = ["create_detector", "DETECTOR_NAMES"]
