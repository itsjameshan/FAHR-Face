"""Detect -> align/crop -> predict -> JSON (spec section 6)."""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.align import align_and_crop
from app.config import Settings
from app.detectors.base import FaceDetection, FaceDetector
from app.models import INPUT_SIZE, FaceModels, Prediction

log = logging.getLogger("fahr.pipeline")

CROP_JPEG_QUALITY = 85


class ImageDecodeError(ValueError):
    """The uploaded bytes are not a decodable image."""


def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes to an upright ``H x W x 3`` uint8 **BGR** array.

    EXIF orientation (phone photos) is applied with ``ImageOps.exif_transpose``
    and the image is converted to RGB (dropping alpha / palettes) before being
    reordered to BGR for the detector.

    Raises:
        ImageDecodeError: for empty, truncated or unsupported data.
    """
    if not data:
        raise ImageDecodeError("empty upload")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im) or im
            rgb = im.convert("RGB")
            arr = np.asarray(rgb, dtype=np.uint8)
    except ImageDecodeError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ImageDecodeError(f"cannot decode image: {type(exc).__name__}") from exc
    except Exception as exc:  # noqa: BLE001 - PIL plugins raise assorted errors
        raise ImageDecodeError(f"cannot decode image: {type(exc).__name__}") from exc
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ImageDecodeError("decoded image has an unexpected shape")
    return np.ascontiguousarray(arr[:, :, ::-1])


def crop_to_data_url(crop_rgb: np.ndarray, size: int = INPUT_SIZE, quality: int = CROP_JPEG_QUALITY) -> str:
    """Resize an RGB crop to ``size x size`` (what the model sees) and encode it as a JPEG data URL."""
    im = Image.fromarray(np.ascontiguousarray(crop_rgb), mode="RGB").resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _ms(start: float, end: float) -> float:
    return round((end - start) * 1000.0, 2)


class Analyzer:
    """Runs the full per-frame pipeline; safe to call from several threads."""

    def __init__(self, detector: FaceDetector, models: FaceModels, settings: Settings) -> None:
        self.detector = detector
        self.models = models
        self.settings = settings

    def analyze(
        self,
        img_bgr: np.ndarray,
        max_faces: Optional[int] = None,
        include_crops: bool = False,
        *,
        decode_ms: float = 0.0,
    ) -> Dict[str, Any]:
        """Analyse one BGR frame and return the spec section 6 response as plain Python types.

        Args:
            img_bgr: ``H x W x 3`` uint8 BGR image.
            max_faces: analyse at most this many faces (highest scores first);
                ``None`` uses ``settings.max_faces``.  Always clamped to
                ``1..settings.max_faces``.
            include_crops: add a 112x112 JPEG data URL per face.
            decode_ms: time the caller spent decoding the image, echoed in ``timing_ms``.
        """
        t_start = time.perf_counter()
        height, width = img_bgr.shape[:2]
        limit = self.settings.max_faces if max_faces is None else max(1, min(int(max_faces), self.settings.max_faces))
        warnings: List[str] = list(self.models.warnings())

        # 1. detect
        t0 = time.perf_counter()
        detections: List[FaceDetection] = list(self.detector.detect(img_bgr))
        t_detect = time.perf_counter()
        detections.sort(key=lambda d: d.score, reverse=True)
        if len(detections) > limit:
            warnings.append(f"{len(detections)} faces detected; analysing the top {limit} of {len(detections)} by score.")
            detections = detections[:limit]

        # 2. align + crop
        kept: List[FaceDetection] = []
        crops: List[np.ndarray] = []
        for det in detections:
            crop = align_and_crop(img_bgr, det.box, det.landmarks)
            if crop is None:
                continue
            kept.append(det)
            crops.append(crop)
        t_align = time.perf_counter()
        if len(kept) < len(detections):
            warnings.append(f"{len(detections) - len(kept)} face(s) dropped: empty crop after alignment.")

        # 3. predict
        predictions: List[Prediction] = self.models.predict(crops) if crops else []
        t_infer = time.perf_counter()

        # 4. assemble
        faces: List[Dict[str, Any]] = []
        for index, (det, crop, pred) in enumerate(zip(kept, crops, predictions)):
            geometry = det.to_dict()  # box, score, landmarks as native Python types
            faces.append(
                {
                    "index": index,
                    "box": geometry["box"],
                    "score": geometry["score"],
                    "landmarks": geometry["landmarks"],
                    "age": None if pred.age is None else float(pred.age),
                    "risk_score": None if pred.risk_score is None else float(pred.risk_score),
                    "crop": crop_to_data_url(crop) if include_crops else None,
                }
            )
        t_end = time.perf_counter()

        return {
            "image": {"width": int(width), "height": int(height)},
            "detector": self.detector.name,
            "faces": faces,
            "models": self.models.summary(),
            "timing_ms": {
                "decode": round(float(decode_ms), 2),
                "detect": _ms(t0, t_detect),
                "align": _ms(t_detect, t_align),
                "infer": _ms(t_align, t_infer),
                "total": round(float(decode_ms) + (t_end - t_start) * 1000.0, 2),
            },
            "warnings": warnings,
        }


__all__ = ["Analyzer", "ImageDecodeError", "decode_image", "crop_to_data_url"]
