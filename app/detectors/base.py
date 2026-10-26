"""Common detection type and detector protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Tuple, runtime_checkable

import numpy as np

LANDMARK_NAMES: Tuple[str, ...] = ("right_eye", "left_eye", "nose", "mouth_right", "mouth_left")


@dataclass
class FaceDetection:
    """One detected face.

    Attributes:
        box: ``(x1, y1, x2, y2)`` in pixels, ints, clipped to the image.
        score: detector confidence in ``[0, 1]``.
        landmarks: five points as ``[x, y]`` floats, named as retina-face does:
            ``right_eye`` / ``mouth_right`` are the *person's* right side, i.e. the
            points that appear on the image's **left**.
    """

    box: Tuple[int, int, int, int]
    score: float
    landmarks: Dict[str, List[float]] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.box
        return {
            "box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
            "score": float(self.score),
            "landmarks": {name: [float(p[0]), float(p[1])] for name, p in self.landmarks.items()},
        }


@runtime_checkable
class FaceDetector(Protocol):
    """A face detector: BGR image in, list of :class:`FaceDetection` out."""

    name: str
    threshold: float

    def detect(self, img_bgr: np.ndarray) -> List[FaceDetection]:  # pragma: no cover - protocol
        ...


__all__ = ["FaceDetection", "FaceDetector", "LANDMARK_NAMES"]
