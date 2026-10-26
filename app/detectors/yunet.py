"""YuNet face detector via ``cv2.FaceDetectorYN`` (OpenCV >= 4.8).

One 230 KB ONNX file, CPU real-time.  The model is fetched once from the OpenCV
model zoo into the cache directory and verified by SHA-256.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

import numpy as np

from app.config import DEFAULT_MODEL_CACHE_DIR
from app.detectors.base import FaceDetection

log = logging.getLogger("fahr.detectors.yunet")

MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
DOWNLOAD_TIMEOUT_S = 60

DEFAULT_THRESHOLD = 0.7
NMS_THRESHOLD = 0.3
TOP_K = 5000


class YuNetModelUnavailable(RuntimeError):
    """The ONNX model is neither cached nor downloadable (or fails verification)."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` (honours ``HTTPS_PROXY`` like urllib does)."""
    req = urllib.request.Request(url, headers={"User-Agent": "fahr-face-live/0.1"})
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as resp, open(dest, "wb") as out:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)


def ensure_model(cache_dir: Optional[str] = None, explicit_path: Optional[str] = None) -> Path:
    """Return a verified path to the YuNet ONNX model, downloading it if needed.

    Args:
        cache_dir: directory for the cached model (default ``~/.cache/fahr-face``).
        explicit_path: use this file instead (``FAHR_YUNET_MODEL``); it is not
            checksum-verified so users can point at other YuNet releases.

    Raises:
        YuNetModelUnavailable: when no usable model can be obtained.
    """
    if explicit_path:
        p = Path(os.path.expanduser(explicit_path))
        if not p.is_file():
            raise YuNetModelUnavailable(f"FAHR_YUNET_MODEL points to a missing file: {p}")
        return p

    cache = Path(os.path.expanduser(cache_dir or DEFAULT_MODEL_CACHE_DIR))
    target = cache / MODEL_FILENAME

    if target.is_file():
        if _sha256(target) == MODEL_SHA256:
            return target
        log.warning("Cached YuNet model %s has an unexpected checksum; re-downloading", target)
        try:
            target.unlink()
        except OSError:
            pass

    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise YuNetModelUnavailable(f"Cannot create model cache directory {cache}: {exc}") from exc

    tmp = target.with_suffix(target.suffix + ".part")
    log.info("Downloading YuNet face detector to %s", target)
    try:
        _download(MODEL_URL, tmp)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise YuNetModelUnavailable(
            f"Could not download the YuNet model from {MODEL_URL}: {exc}. "
            f"Download it manually to {target} or set FAHR_YUNET_MODEL."
        ) from exc

    digest = _sha256(tmp)
    if digest != MODEL_SHA256:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise YuNetModelUnavailable(
            f"Downloaded YuNet model failed verification (sha256 {digest}, expected {MODEL_SHA256})"
        )
    os.replace(tmp, target)
    return target


def _clip_int(value: float, lo: int, hi: int) -> int:
    return max(lo, min(int(value), hi))


class YuNetDetector:
    """``cv2.FaceDetectorYN`` wrapper producing :class:`FaceDetection` objects.

    The detector object is not thread-safe, so :meth:`detect` is serialised.
    """

    name = "yunet"

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: Optional[float] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        import cv2  # local import keeps `app` importable without OpenCV

        self._cv2 = cv2
        self.threshold = float(DEFAULT_THRESHOLD if threshold is None else threshold)
        self.model_path = str(ensure_model(cache_dir=cache_dir, explicit_path=model_path))
        self._lock = threading.Lock()
        self._input_size = (320, 320)

        # Silence OpenCV's harmless "Targets are not supported by the new graph engine" warning.
        log_mod = getattr(getattr(cv2, "utils", None), "logging", None)
        previous = None
        if log_mod is not None:
            try:
                previous = log_mod.getLogLevel()
                log_mod.setLogLevel(log_mod.LOG_LEVEL_ERROR)
            except Exception:  # pragma: no cover - depends on the OpenCV build
                previous = None
        try:
            self._detector = cv2.FaceDetectorYN.create(
                self.model_path, "", self._input_size, self.threshold, NMS_THRESHOLD, TOP_K
            )
        finally:
            if previous is not None:
                try:
                    log_mod.setLogLevel(previous)
                except Exception:  # pragma: no cover
                    pass

    def detect(self, img_bgr: np.ndarray) -> List[FaceDetection]:
        if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError("detect() expects an H x W x 3 BGR image")
        h, w = img_bgr.shape[:2]
        if h < 1 or w < 1:
            return []
        if img_bgr.dtype != np.uint8:
            img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)
        if not img_bgr.flags["C_CONTIGUOUS"]:
            img_bgr = np.ascontiguousarray(img_bgr)

        with self._lock:
            if (w, h) != self._input_size:
                self._detector.setInputSize((w, h))
                self._input_size = (w, h)
            _, rows = self._detector.detect(img_bgr)

        if rows is None or len(rows) == 0:
            return []

        detections: List[FaceDetection] = []
        for row in np.asarray(rows, dtype=np.float64):
            # Row layout: x, y, w, h, 5 x (lx, ly), score.
            x, y, bw, bh = row[0:4]
            points = row[4:14].reshape(5, 2)
            score = float(row[14])

            x1 = _clip_int(x, 0, w)
            y1 = _clip_int(y, 0, h)
            x2 = _clip_int(x + bw, 0, w)
            y2 = _clip_int(y + bh, 0, h)
            if x2 <= x1 or y2 <= y1:
                continue

            # Name landmarks by image position (retina-face convention: the
            # person's right eye is the one on the image's left).
            eyes = sorted((points[0], points[1]), key=lambda p: p[0])
            mouth = sorted((points[3], points[4]), key=lambda p: p[0])
            landmarks = {
                "right_eye": [float(eyes[0][0]), float(eyes[0][1])],
                "left_eye": [float(eyes[1][0]), float(eyes[1][1])],
                "nose": [float(points[2][0]), float(points[2][1])],
                "mouth_right": [float(mouth[0][0]), float(mouth[0][1])],
                "mouth_left": [float(mouth[1][0]), float(mouth[1][1])],
            }
            detections.append(FaceDetection(box=(x1, y1, x2, y2), score=score, landmarks=landmarks))

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections


__all__ = [
    "YuNetDetector",
    "YuNetModelUnavailable",
    "ensure_model",
    "MODEL_FILENAME",
    "MODEL_URL",
    "MODEL_SHA256",
    "DEFAULT_THRESHOLD",
]
