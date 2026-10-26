"""FastAPI server: health schema, analyze contract, error codes, headers."""

from __future__ import annotations

import base64
import io
import math

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")

from PIL import Image  # noqa: E402

from tests.fixtures import require_fixture  # noqa: E402

API_HEADERS = {
    "cache-control": "no-store",
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
}


def _jpeg_bytes(array_rgb: np.ndarray, **save_kwargs) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array_rgb).save(buf, format="JPEG", **save_kwargs)
    return buf.getvalue()


def _post(client, data: bytes, **params):
    return client.post("/api/analyze", files={"image": ("frame.jpg", data, "image/jpeg")}, params=params)


# ------------------------------------------------------------------ health


def test_health_schema(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"status", "version", "device", "detector", "models", "limits"}
    assert body["status"] in {"ok", "degraded"}
    assert body["version"] == "0.1.0"
    assert body["device"] in {"cpu", "cuda", "mps"}
    assert set(body["detector"]) == {"name", "threshold", "available", "error"}
    assert body["detector"]["name"] == "yunet"
    assert body["detector"]["available"] is True
    assert body["detector"]["error"] is None
    assert isinstance(body["detector"]["threshold"], float)
    for key in ("age", "survival"):
        assert set(body["models"][key]) == {"available", "weights", "checkpoint", "error"}
    assert body["models"]["age"]["weights"] == "random"
    assert body["models"]["age"]["available"] is True
    assert body["models"]["survival"]["weights"] == "none"
    assert body["models"]["survival"]["available"] is False
    assert body["models"]["survival"]["error"] == (
        "Checkpoint not found. FAHR-FaceSurvival weights are not publicly released."
    )
    assert body["limits"] == {"max_faces": 5, "max_upload_bytes": 1_000_000}
    # random weights -> never "ok"
    assert body["status"] == "degraded"


def test_api_headers(client):
    r = client.get("/api/health")
    for name, value in API_HEADERS.items():
        assert r.headers.get(name) == value, name
    assert r.headers.get("permissions-policy") == "camera=(self)"


def test_root_serves_ui_or_explains(client):
    r = client.get("/")
    assert r.headers.get("permissions-policy") == "camera=(self)"
    if r.status_code == 200:
        assert r.headers["content-type"].startswith("text/html")
    else:
        assert r.status_code == 404
        assert r.json() == {"detail": "UI not built"}


# ----------------------------------------------------------------- analyze


def test_analyze_one_face_with_age(client):
    data = require_fixture("obama.jpg").read_bytes()
    r = _post(client, data)
    assert r.status_code == 200, r.text
    for name, value in API_HEADERS.items():
        assert r.headers.get(name) == value, name
    body = r.json()
    assert set(body) == {"image", "detector", "faces", "models", "timing_ms", "warnings"}
    assert body["image"] == {"width": 910, "height": 1137}
    assert body["detector"] == "yunet"
    assert len(body["faces"]) == 1
    face = body["faces"][0]
    assert set(face) == {"index", "box", "score", "landmarks", "age", "risk_score", "crop"}
    assert face["index"] == 0
    assert set(face["box"]) == {"x1", "y1", "x2", "y2"}
    assert all(isinstance(v, int) for v in face["box"].values())
    assert 0 <= face["box"]["x1"] < face["box"]["x2"] <= 910
    assert 0 <= face["box"]["y1"] < face["box"]["y2"] <= 1137
    assert isinstance(face["score"], float) and 0.7 <= face["score"] <= 1.0
    assert set(face["landmarks"]) == {"right_eye", "left_eye", "nose", "mouth_right", "mouth_left"}
    assert face["landmarks"]["right_eye"][0] < face["landmarks"]["left_eye"][0]
    assert isinstance(face["age"], float) and math.isfinite(face["age"])
    assert face["risk_score"] is None
    assert face["crop"] is None  # include_crops defaults to False
    assert body["models"]["age"] == {"available": True, "weights": "random", "reason": body["models"]["age"]["reason"]}
    assert body["models"]["survival"]["available"] is False
    assert body["models"]["survival"]["weights"] == "none"
    assert "not publicly released" in body["models"]["survival"]["reason"]
    assert set(body["timing_ms"]) == {"decode", "detect", "align", "infer", "total"}
    assert all(isinstance(v, float) and v >= 0 for v in body["timing_ms"].values())
    assert body["timing_ms"]["total"] > 0
    assert any("random" in w.lower() for w in body["warnings"])


def test_analyze_two_faces_sorted_by_score(client):
    data = require_fixture("two_people.jpg").read_bytes()
    body = _post(client, data).json()
    assert len(body["faces"]) == 2
    assert [f["index"] for f in body["faces"]] == [0, 1]
    assert body["faces"][0]["score"] >= body["faces"][1]["score"]


def test_max_faces_is_clamped(client):
    data = require_fixture("two_people.jpg").read_bytes()
    body = _post(client, data, max_faces=1).json()
    assert len(body["faces"]) == 1
    assert any("1 of 2" in w or "top 1" in w.lower() for w in body["warnings"])
    # Out-of-range values clamp instead of failing.
    assert len(_post(client, data, max_faces=0).json()["faces"]) == 1
    assert len(_post(client, data, max_faces=999).json()["faces"]) == 2


def test_blank_image_gives_no_faces(client):
    blank = np.full((240, 320, 3), 128, dtype=np.uint8)
    r = _post(client, _jpeg_bytes(blank))
    assert r.status_code == 200
    body = r.json()
    assert body["faces"] == []
    assert body["image"] == {"width": 320, "height": 240}


def test_garbage_bytes_give_400(client):
    r = _post(client, b"this is not an image at all" * 10)
    assert r.status_code == 400
    assert "detail" in r.json()


def test_empty_upload_gives_400(client):
    r = _post(client, b"")
    assert r.status_code == 400


def test_oversized_upload_gives_413(client):
    r = _post(client, b"\xff" * 1_000_001)
    assert r.status_code == 413
    assert "detail" in r.json()


def test_uploads_within_the_limit_never_touch_the_disk(monkeypatch):
    """Starlette spools multipart files above 1 MB to disk unless the threshold is raised."""
    import tempfile

    from fastapi.testclient import TestClient
    from starlette.formparsers import MultiPartParser

    from app.config import Settings
    from app.server import create_app

    limit = 3_000_000
    settings = Settings(
        allow_random_weights=False,
        age_checkpoint="/nonexistent/age.pth",
        survival_checkpoint="/nonexistent/surv.pth",
        max_upload_bytes=limit,
        device="cpu",
    )

    rollovers = []

    def spy_rollover(self):
        rollovers.append(self._max_size)
        return original_rollover(self)

    original_rollover = tempfile.SpooledTemporaryFile.rollover
    monkeypatch.setattr(tempfile.SpooledTemporaryFile, "rollover", spy_rollover)

    with TestClient(create_app(settings)) as c:
        assert MultiPartParser.spool_max_size > limit
        # A 1.5 MB JPEG: above Starlette's default 1 MB spool size, below our limit.
        noise = np.random.default_rng(0).integers(0, 256, size=(1200, 1600, 3), dtype=np.uint8)
        data = _jpeg_bytes(noise, quality=95)
        assert 1_048_576 < len(data) <= limit
        r = _post(c, data)
        assert r.status_code == 200
        assert r.json()["image"] == {"width": 1600, "height": 1200}
    assert rollovers == []


def test_include_crops_returns_data_urls(client):
    data = require_fixture("obama.jpg").read_bytes()
    body = _post(client, data, include_crops="true").json()
    assert len(body["faces"]) == 1
    crop = body["faces"][0]["crop"]
    prefix = "data:image/jpeg;base64,"
    assert crop.startswith(prefix)
    img = Image.open(io.BytesIO(base64.b64decode(crop[len(prefix):])))
    assert img.format == "JPEG"
    assert img.size == (112, 112)
    assert img.mode == "RGB"


def test_exif_rotated_upload_is_analysed_upright(client):
    """Phone photos carry EXIF orientation; the reported size must be the upright one."""
    blank = np.full((240, 320, 3), 90, dtype=np.uint8)  # landscape 320x240 ...
    exif = Image.Exif()
    exif[274] = 6  # ... tagged "rotate 90 CW" -> upright is 240x320
    data = _jpeg_bytes(blank, exif=exif.tobytes())
    body = _post(client, data).json()
    assert body["image"] == {"width": 240, "height": 320}


def test_png_upload_is_accepted(client):
    buf = io.BytesIO()
    Image.fromarray(np.zeros((64, 48, 3), dtype=np.uint8)).save(buf, format="PNG")
    r = client.post("/api/analyze", files={"image": ("f.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200
    assert r.json()["image"] == {"width": 48, "height": 64}


def test_missing_image_field_is_rejected(client):
    r = client.post("/api/analyze", files={"photo": ("f.jpg", b"123", "image/jpeg")})
    assert r.status_code == 422


# --------------------------------------------------------- degraded startup


def test_detector_failure_gives_503_but_server_starts():
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.server import create_app

    settings = Settings(
        detector="no-such-detector",
        allow_random_weights=False,
        age_checkpoint="/nonexistent/age.pth",
        survival_checkpoint="/nonexistent/surv.pth",
        device="cpu",
    )
    with TestClient(create_app(settings)) as c:
        health = c.get("/api/health").json()
        assert health["status"] == "degraded"
        assert health["detector"]["available"] is False
        assert "no-such-detector" in health["detector"]["error"]
        assert health["models"]["age"]["weights"] == "none"
        blank = np.full((240, 320, 3), 128, dtype=np.uint8)
        r = _post(c, _jpeg_bytes(blank))
        assert r.status_code == 503


# ------------------------------------------------------------ settings


def test_settings_from_env():
    from app.config import Settings

    defaults = Settings.from_env({})
    assert defaults == Settings()
    assert defaults.age_checkpoint == "checkpoints/fahr_faceage.pth"
    assert defaults.survival_checkpoint == "checkpoints/fahr_facesurvival.pth"
    assert defaults.detector == "auto" and defaults.detection_threshold is None
    assert defaults.device == "auto" and defaults.allow_random_weights is False
    assert defaults.max_faces == 5 and defaults.max_upload_bytes == 8388608
    assert defaults.model_cache_dir == "~/.cache/fahr-face" and defaults.yunet_model is None
    assert (defaults.host, defaults.port) == ("127.0.0.1", 8000)

    custom = Settings.from_env(
        {
            "FAHR_AGE_CHECKPOINT": "/w/age.pth",
            "FAHR_SURVIVAL_CHECKPOINT": "/w/surv.pth",
            "FAHR_DETECTOR": "RetinaFace",
            "FAHR_DETECTION_THRESHOLD": "0.55",
            "FAHR_DEVICE": "CPU",
            "FAHR_ALLOW_RANDOM_WEIGHTS": "1",
            "FAHR_MAX_FACES": "3",
            "FAHR_MAX_UPLOAD_BYTES": "4096",
            "FAHR_MODEL_CACHE_DIR": "/tmp/cache",
            "FAHR_YUNET_MODEL": "/w/yunet.onnx",
            "FAHR_HOST": "0.0.0.0",
            "FAHR_PORT": "9000",
        }
    )
    assert custom.detector == "retinaface" and custom.device == "cpu"
    assert custom.detection_threshold == pytest.approx(0.55)
    assert custom.allow_random_weights is True
    assert (custom.max_faces, custom.max_upload_bytes) == (3, 4096)
    assert custom.model_cache_dir == "/tmp/cache" and custom.yunet_model == "/w/yunet.onnx"
    assert (custom.host, custom.port) == ("0.0.0.0", 9000)

    with pytest.raises(ValueError):
        Settings.from_env({"FAHR_MAX_FACES": "zero"})
    with pytest.raises(ValueError):
        Settings.from_env({"FAHR_ALLOW_RANDOM_WEIGHTS": "maybe"})
    with pytest.raises(ValueError):
        Settings.from_env({"FAHR_PORT": "0"})


# ----------------------------------------------------------- unit: decode


def test_decode_image_helper():
    from app.pipeline import ImageDecodeError, decode_image

    rgb = np.zeros((10, 20, 3), dtype=np.uint8)
    rgb[..., 0] = 255  # pure red in RGB
    bgr = decode_image(_jpeg_bytes(rgb, quality=100))
    assert bgr.shape == (10, 20, 3)
    assert bgr.dtype == np.uint8
    assert bgr.flags["C_CONTIGUOUS"]
    assert bgr[0, 0, 2] > 200 and bgr[0, 0, 0] < 50  # red landed in the B,G,R -> R channel
    with pytest.raises(ImageDecodeError):
        decode_image(b"nope")
    with pytest.raises(ImageDecodeError):
        decode_image(b"")
