"""YuNet detector: face counts on known photos, landmark naming, blank frames."""

from __future__ import annotations

import numpy as np
import pytest

from tests.fixtures import require_fixture

cv2 = pytest.importorskip("cv2")


@pytest.fixture(scope="module")
def detector():
    from app.detectors.yunet import YuNetDetector, YuNetModelUnavailable

    try:
        return YuNetDetector()
    except YuNetModelUnavailable as exc:  # offline and not cached
        pytest.skip(f"YuNet model unavailable: {exc}")


def _load(name: str) -> np.ndarray:
    img = cv2.imread(str(require_fixture(name)))
    assert img is not None
    return img


def _assert_face_shape(face, img):
    from app.detectors.base import FaceDetection

    assert isinstance(face, FaceDetection)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = face.box
    assert all(isinstance(v, int) for v in face.box)
    assert 0 <= x1 < x2 <= w
    assert 0 <= y1 < y2 <= h
    assert isinstance(face.score, float) and 0.0 <= face.score <= 1.0
    assert set(face.landmarks) == {"right_eye", "left_eye", "nose", "mouth_right", "mouth_left"}
    for point in face.landmarks.values():
        assert len(point) == 2
        assert all(isinstance(v, float) for v in point)
    # retina-face naming: "right_eye" is the eye on the image's left.
    assert face.landmarks["right_eye"][0] < face.landmarks["left_eye"][0]
    assert face.landmarks["mouth_right"][0] < face.landmarks["mouth_left"][0]


@pytest.mark.parametrize(
    "name, expected",
    [("obama.jpg", 1), ("two_people.jpg", 2), ("biden.jpg", 1)],
)
def test_face_counts(detector, name, expected):
    img = _load(name)
    faces = detector.detect(img)
    assert len(faces) == expected
    for face in faces:
        _assert_face_shape(face, img)


def test_faces_sorted_by_score_and_landmarks_inside_box_neighbourhood(detector):
    img = _load("two_people.jpg")
    faces = detector.detect(img)
    scores = [f.score for f in faces]
    assert scores == sorted(scores, reverse=True)
    for face in faces:
        x1, y1, x2, y2 = face.box
        for px, py in face.landmarks.values():
            # Landmarks lie in or very near the box.
            assert x1 - 20 <= px <= x2 + 20
            assert y1 - 20 <= py <= y2 + 20


def test_blank_frame_gives_no_faces(detector):
    blank = np.full((240, 320, 3), 128, dtype=np.uint8)
    assert detector.detect(blank) == []


def test_input_size_changes_between_frames(detector):
    """The detector must adapt to a new frame size without errors."""
    big = _load("obama.jpg")
    small = cv2.resize(big, (big.shape[1] // 2, big.shape[0] // 2))
    assert len(detector.detect(big)) == 1
    assert len(detector.detect(small)) == 1
    assert len(detector.detect(big)) == 1


def test_detector_metadata(detector):
    assert detector.name == "yunet"
    assert detector.threshold == pytest.approx(0.7)


def test_factory():
    from app.detectors.factory import create_detector
    from app.detectors.yunet import YuNetDetector, YuNetModelUnavailable

    try:
        auto = create_detector("auto")
    except YuNetModelUnavailable as exc:
        pytest.skip(f"YuNet model unavailable: {exc}")
    assert isinstance(auto, YuNetDetector)
    explicit = create_detector("yunet", threshold=0.5)
    assert explicit.threshold == pytest.approx(0.5)
    with pytest.raises(ValueError):
        create_detector("does-not-exist")


def test_model_download_verifies_checksum(tmp_path, monkeypatch):
    from app.detectors import yunet

    bad = tmp_path / yunet.MODEL_FILENAME
    bad.write_bytes(b"not an onnx file")

    def fake_download(url, dest):  # noqa: ARG001
        dest.write_bytes(b"still not an onnx file")

    monkeypatch.setattr(yunet, "_download", fake_download)
    with pytest.raises(yunet.YuNetModelUnavailable):
        yunet.ensure_model(cache_dir=str(tmp_path))
    # Neither the stale cached file nor the bad download may be left in place.
    assert not bad.exists()
    assert not list(tmp_path.iterdir())


def test_model_download_success_path(tmp_path, monkeypatch):
    """A download with the right checksum is moved into the cache and reused."""
    from app.detectors import yunet

    real = yunet.ensure_model()  # cached or downloaded for real; skip if neither works
    payload = real.read_bytes()
    calls = []

    def fake_download(url, dest):
        calls.append(url)
        dest.write_bytes(payload)

    monkeypatch.setattr(yunet, "_download", fake_download)
    got = yunet.ensure_model(cache_dir=str(tmp_path))
    assert got == tmp_path / yunet.MODEL_FILENAME
    assert calls == [yunet.MODEL_URL]
    assert yunet.ensure_model(cache_dir=str(tmp_path)) == got
    assert calls == [yunet.MODEL_URL]  # second call served from cache


def test_explicit_model_path_missing_raises(tmp_path):
    from app.detectors import yunet

    with pytest.raises(yunet.YuNetModelUnavailable):
        yunet.ensure_model(cache_dir=str(tmp_path), explicit_path=str(tmp_path / "missing.onnx"))
