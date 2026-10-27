# FAHR-Face Live: lightweight webcam demo — design spec

**Date:** 2026-10-26
**Status:** proposed (awaiting go-ahead)

## 1. Intent

### 1.1 Request

> "Can you deploy this with a UI and a camera which can recognize human face and survival?"
> "When I say camera, I meant a webcam or something light."

### 1.2 Understanding

A small, local web app on top of this repository that:

1. opens the user's webcam in the browser,
2. detects human faces in the live video (detection / localisation, not identification),
3. preprocesses each face the way the paper did (detect → eye-based alignment → crop → 112×112 → ImageNet normalisation),
4. runs **FAHR-FaceAge** (released weights) → predicted face age,
5. runs **FAHR-FaceSurvival** → relative risk score **only when the user has that checkpoint**; otherwise shows "not available" (the weights are not publicly released),
6. draws the results as a live overlay and runs with one command: `python -m app`.

"Light" means: no TensorFlow, no heavy container image, no build step. Dependencies are torch, torchvision, transformers, fastapi, uvicorn, opencv-python-headless, pillow, numpy.

### 1.3 Assumptions

| # | Assumption | Why |
|---|------------|-----|
| A1 | Survival = FAHR-FaceSurvival risk score, shown only if `checkpoints/fahr_facesurvival.pth` exists. | README: survival weights are not distributed. |
| A2 | Age (FAHR-FaceAge) is the primary displayed indicator. | It is the only released downstream model. |
| A3 | Face detection uses OpenCV's bundled YuNet detector (one 230 KB ONNX file, ~30 fps on CPU) instead of the TensorFlow RetinaFace package used by `run.py`. The alignment/crop geometry is a direct port of retina-face 0.0.17, so the age model sees the same kind of crop. An optional `FAHR_DETECTOR=retinaface` switch uses the original package if the user has it installed. | Keeps the demo light while preserving preprocessing fidelity. |
| A4 | This sandbox cannot reach huggingface.co, so the real checkpoint cannot be downloaded here. The pipeline is verified with randomly initialised weights of the same architecture; the user downloads `fahr_faceage.pth` on their machine. | Network policy (403). |
| A5 | The webcam runs in the user's browser and the server on the same machine (`localhost`), which is a secure context for `getUserMedia`. | Browser security model. |

## 2. Non-goals

- Face identification of individuals; storing frames or predictions; training.
- Calibrated survival probabilities (the survival head is an uncalibrated ranking score).
- Public hosting / production deployment.

## 3. Architecture

```
Browser (one HTML page, vanilla JS, no CDN)         Local server (FastAPI + uvicorn)
getUserMedia → <video> → canvas (≤640 px) → JPEG ──POST /api/analyze──► decode → YuNet detect
overlay boxes + "Age 42 · Risk n/a"          ◄──────── JSON ─────────── → align/crop (retina-face port)
per-face cards, smoothing, snapshot/upload                                → 112×112 → ViT age (+ risk)
                                                                         GET /api/health, GET /
```

One HTTP POST per frame, issued only after the previous response arrives (adaptive rate).

## 4. Backend (`app/` package at repository root)

The package adds the repository root to `sys.path` so `ViTForAgeEstimation.py` and `ViTForSurvivalAnalysis.py` are reused unchanged.

| File | Responsibility |
|------|----------------|
| `app/__init__.py`, `app/__main__.py` | version; `python -m app` runs uvicorn with host/port from env. |
| `app/config.py` | `Settings` from environment variables (§4.1). |
| `app/align.py` | Exact port of retina-face 0.0.17 `extract_faces(align=True)` geometry (§4.2). numpy + PIL only. |
| `app/detectors/base.py` | `FaceDetection` (box x1,y1,x2,y2; score; landmarks `right_eye, left_eye, nose, mouth_right, mouth_left` as `[x, y]`, retina-face naming where `right_eye` appears on the image's left) and the `FaceDetector` protocol. |
| `app/detectors/yunet.py` | `cv2.FaceDetectorYN` wrapper. Downloads `face_detection_yunet_2023mar.onnx` (sha256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`) from `https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx` to the cache dir on first run; maps landmarks to retina-face naming by image position; default threshold 0.7. |
| `app/detectors/retinaface.py` | Optional thin wrapper around `retinaface.RetinaFace.detect_faces` (only if that package is installed). |
| `app/detectors/factory.py` | `create_detector("auto"|"yunet"|"retinaface")`; `auto` = YuNet. |
| `app/models.py` | Loads age and survival models with the exact configs from `run.py`; missing checkpoint → state `none` (or `random` with `FAHR_ALLOW_RANDOM_WEIGHTS=1`) with a reason, never crashes at startup. `predict(crops_rgb)` applies `Resize((112,112)) → ToTensor → Normalize(ImageNet)` and runs one batched forward pass per model (handles `.squeeze()` for batch size 1). Device auto: cuda → mps → cpu. |
| `app/pipeline.py` | `Analyzer.analyze(img_bgr, max_faces, include_crops)`: detect → top-N by score → align/crop → predict → JSON (§6) with timings and optional base64 crop thumbnails. |
| `app/server.py` | FastAPI: `GET /`, `/static/*`, `GET /api/health`, `POST /api/analyze`. Limits upload size (413), rejects undecodable images (400), 503 if detector failed to load. Inference in a threadpool. `Cache-Control: no-store` on API; nothing stored or logged about faces. |

### 4.1 Settings

| Variable | Default | Meaning |
|----------|---------|---------|
| `FAHR_AGE_CHECKPOINT` | `checkpoints/fahr_faceage.pth` | Age checkpoint. |
| `FAHR_SURVIVAL_CHECKPOINT` | `checkpoints/fahr_facesurvival.pth` | Survival checkpoint (optional). |
| `FAHR_DETECTOR` | `auto` | `auto`/`yunet`/`retinaface`. |
| `FAHR_DETECTION_THRESHOLD` | detector default | Score threshold. |
| `FAHR_DEVICE` | `auto` | `auto`/`cpu`/`cuda`/`mps`. |
| `FAHR_ALLOW_RANDOM_WEIGHTS` | `0` | Dev/test only: random init when a checkpoint is missing; every response carries a warning. |
| `FAHR_MAX_FACES` | `5` | Faces analysed per frame. |
| `FAHR_MAX_UPLOAD_BYTES` | `8388608` | Upload limit. |
| `FAHR_MODEL_CACHE_DIR` | `~/.cache/fahr-face` | Detector weight cache. |
| `FAHR_YUNET_MODEL` | (unset) | Explicit YuNet ONNX path. |
| `FAHR_HOST` / `FAHR_PORT` | `127.0.0.1` / `8000` | Bind address. |

### 4.2 Alignment port — exact semantics

Given BGR image `img`, facial area `(x1, y1, x2, y2)` and landmarks:

1. `left_eye := landmarks["right_eye"]`, `right_eye := landmarks["left_eye"]` (retina-face swaps names: its names are the person's eyes, the procedure wants image-left/right).
2. `direction = -1` if `left_eye_y > right_eye_y` (third point `(right_eye_x, left_eye_y)`), else `+1` (third point `(left_eye_x, right_eye_y)`).
3. `a = |left_eye − p3|`, `b = |right_eye − p3|`, `c = |right_eye − left_eye|`; if `b == 0 or c == 0` → no rotation.
4. `cos_a = clamp((b² + c² − a²)/(2bc), −1, 1)`; `angle = degrees(arccos cos_a)`; if `direction == −1`: `angle = 90 − angle`.
5. `rotated = np.array(PIL.Image.fromarray(img).rotate(direction * angle))` (PIL defaults: NEAREST, no expand, image centre).
6. Rotate the box about the image centre as `rotate_facial_area` does (`int()` truncation, clamp to image).
7. Crop and return RGB.

A test vendors the original two retina-face functions as an oracle and asserts array equality.

## 5. Frontend (`app/static/index.html`, `app.js`, `styles.css`)

- Header with status chips from `/api/health`: detector, device, age model (ready / random weights / missing), survival model (ready / not available).
- Stage: `<video>` + `<canvas>` overlay; boxes, optional landmark dots, label "Age 42 · Risk n/a". Mirror via CSS on the video only; overlay x-coordinates mirrored arithmetically so text stays readable.
- Side panel: one card per face with smoothed age, risk score (or "not available" + why), confidence, and the 112×112 aligned crop ("what the model sees").
- Controls: Start/Stop camera, camera selector, Continuous ⇄ Snapshot, Upload image, mirror and landmarks toggles.
- Banners: missing checkpoint (shows the download command), random-weights warning, survival unavailable, insecure context, camera permission errors.
- Consent note before the camera starts: frames go only to this local server and are not stored; research use only, not medical advice.
- Loop: capture ≤640 px JPEG → POST → draw → next; back-off on errors. Faces matched across frames by IoU ≥ 0.3; EMA (α = 0.3) on age and risk.
- Light/dark via `prefers-color-scheme`; works at phone width; buttons labelled; status `aria-live`.

## 6. API contract

`GET /api/health` →

```json
{"status": "ok|degraded", "version": "0.1.0", "device": "cpu",
 "detector": {"name": "yunet", "threshold": 0.7, "available": true, "error": null},
 "models": {"age": {"available": true, "weights": "checkpoint|random|none", "checkpoint": "checkpoints/fahr_faceage.pth", "error": null},
            "survival": {"available": false, "weights": "none", "checkpoint": "checkpoints/fahr_facesurvival.pth",
                         "error": "Checkpoint not found. FAHR-FaceSurvival weights are not publicly released."}},
 "limits": {"max_faces": 5, "max_upload_bytes": 8388608}}
```

`POST /api/analyze?include_crops=<bool>&max_faces=<int>` with multipart field `image` →

```json
{"image": {"width": 640, "height": 480}, "detector": "yunet",
 "faces": [{"index": 0, "box": {"x1": 100, "y1": 80, "x2": 300, "y2": 330}, "score": 0.98,
            "landmarks": {"right_eye": [150.2, 160.1], "left_eye": [250.7, 158.9], "nose": [200, 220],
                          "mouth_right": [160, 280], "mouth_left": [240, 282]},
            "age": 42.3, "risk_score": null, "crop": "data:image/jpeg;base64,..."}],
 "models": {"age": {"available": true, "weights": "checkpoint"},
            "survival": {"available": false, "weights": "none", "reason": "..."}},
 "timing_ms": {"decode": 3.1, "detect": 20.5, "align": 1.2, "infer": 85.0, "total": 110.0},
 "warnings": []}
```

Errors: `400` undecodable image, `413` too large, `503` detector unavailable.

## 7. Running it

- `requirements-app.txt` (torch, torchvision, transformers, fastapi, uvicorn[standard], python-multipart, pillow, opencv-python-headless, numpy).
- `scripts/download_checkpoints.sh`: curl with resume from Hugging Face into `checkpoints/`.
- Run: `pip install -r requirements-app.txt && python -m app`, then open `http://localhost:8000`.
- README gets a short *Live webcam demo* section (quick start, what is shown, survival availability, env vars).
- Optional small `Dockerfile` (CPU torch, YuNet only) only if requested.

## 8. Testing

| Test | Proves |
|------|--------|
| `tests/test_align.py` | Port equals the vendored retina-face oracle (no rotation, both directions, degenerate eyes, border boxes). |
| `tests/test_yunet.py` | Detector finds one face in a portrait and two in a two-person photo; landmark naming order. Fixtures downloaded from public GitHub repos into `tests/fixtures/.cache/`; skipped offline. |
| `tests/test_models.py` | Random-weights mode gives finite floats; batch sizes 1 and N; missing checkpoint → `none` with reason; survival absent → `risk_score: null`. |
| `tests/test_server.py` | Health schema; analyze → one face with age; blank image → `[]`; garbage → 400; oversized → 413; crops are data URLs. |
| `scripts/e2e_fake_camera.mjs` | Playwright launches Chromium with a fake webcam (Y4M built from a portrait), accepts the consent note, and waits for a face card with an age; saves a screenshot. Run here with random weights. |

## 9. Risks

- YuNet boxes differ slightly from RetinaFace boxes → may bias ages by a few years; the active detector is always shown and `FAHR_DETECTOR=retinaface` is available.
- Random-weights mode must never be mistaken for real output → red banner, warning in every response, off by default.
- Survival score misread as a probability → labelled "relative risk score (uncalibrated)", shown only with the user's own checkpoint.
