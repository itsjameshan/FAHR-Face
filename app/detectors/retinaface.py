"""Optional wrapper around the original ``retina-face`` package (TensorFlow).

Only usable when ``pip install retina-face`` was done by the user; the import is
lazy so the rest of the app never touches TensorFlow.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional

import numpy as np

from app.detectors.base import LANDMARK_NAMES, FaceDetection

log = logging.getLogger("fahr.detectors.retinaface")

DEFAULT_THRESHOLD = 0.9


class RetinaFaceDetector:
    """``retinaface.RetinaFace.detect_faces`` as a :class:`FaceDetector`."""

    name = "retinaface"

    def __init__(self, threshold: Optional[float] = None) -> None:
        try:
            from retinaface import RetinaFace  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "FAHR_DETECTOR=retinaface requires the optional 'retina-face' package "
                "(pip install retina-face), which pulls in TensorFlow."
            ) from exc
        self._rf = RetinaFace
        self.threshold = float(DEFAULT_THRESHOLD if threshold is None else threshold)
        self._lock = threading.Lock()
        # Build the TF model eagerly so the first request is not slow and errors surface at startup.
        self._rf.build_model()

    def detect(self, img_bgr: np.ndarray) -> List[FaceDetection]:
        if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError("detect() expects an H x W x 3 BGR image")
        h, w = img_bgr.shape[:2]
        with self._lock:
            result = self._rf.detect_faces(img_bgr, threshold=self.threshold)
        if not isinstance(result, dict):
            return []

        detections: List[FaceDetection] = []
        for identity in result.values():
            area = identity.get("facial_area")
            lms = identity.get("landmarks") or {}
            if area is None or len(area) != 4:
                continue
            x1 = max(0, min(int(area[0]), w))
            y1 = max(0, min(int(area[1]), h))
            x2 = max(0, min(int(area[2]), w))
            y2 = max(0, min(int(area[3]), h))
            if x2 <= x1 or y2 <= y1:
                continue
            landmarks = {}
            for name in LANDMARK_NAMES:
                point = lms.get(name)
                if point is None:
                    continue
                landmarks[name] = [float(point[0]), float(point[1])]
            detections.append(
                FaceDetection(box=(x1, y1, x2, y2), score=float(identity.get("score", 0.0)), landmarks=landmarks)
            )
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections


__all__ = ["RetinaFaceDetector", "DEFAULT_THRESHOLD"]
