"""Face detectors producing retina-face style detections (box, score, 5 landmarks)."""

from app.detectors.base import FaceDetection, FaceDetector
from app.detectors.factory import create_detector

__all__ = ["FaceDetection", "FaceDetector", "create_detector"]
