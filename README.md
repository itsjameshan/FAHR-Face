# FAHR-Face

Code accompanying **Foundation artificial intelligence models for health recognition by use of face photographs (FAHR-Face): a model development and validation study**, Haugg, Lee, and colleagues, *The Lancet Digital Health* (2026). [Paper](https://doi.org/10.1016/j.landig.2026.101056) · [Model checkpoints](https://huggingface.co/fridolin22/FAHR-Face)

This repository contains the implementation of the FAHR (Foundation Artificial Intelligence Models for Health) models as described in the associated publication. The codebase includes three main components:

1. **FAHR-Face** - Foundation model trained on facial images using a Masked Autoencoder approach
2. **FAHR-FaceAge** - Age estimation model built on top of the foundation model
3. **FAHR-FaceSurvival** - Survival analysis model for health risk prediction

**Released weights: FAHR-Face and FAHR-FaceAge only.** Both are publicly available on [Hugging Face](https://huggingface.co/fridolin22/FAHR-Face); FAHR-FaceSurvival weights are not publicly released.

**FAHR-FaceAge requires cropped and aligned face images before inference.** See [Face Preprocessing](#face-preprocessing).

## Repository Structure

```
FAHR-Face/
├── run.py                        # Main script to run inference with trained models
├── ViTForAgeEstimation.py        # Age estimation model architecture
├── ViTForSurvivalAnalysis.py     # Survival analysis model architecture
├── facefoundation.yml            # Miniforge environment configuration
├── requirements-app.txt          # Dependencies of the live webcam demo (CPU PyTorch, no TensorFlow)
├── app/                          # Live webcam demo: local FastAPI server, run with `python -m app`
│   ├── __main__.py               # Entry point; configured through FAHR_* environment variables
│   ├── config.py                 # Settings read from the environment
│   ├── align.py                  # Face alignment and crop geometry ported from retina-face 0.0.17
│   ├── detectors/                # Face detectors: OpenCV YuNet (default), optional RetinaFace
│   ├── models.py                 # Loads FAHR-FaceAge / FAHR-FaceSurvival and runs inference
│   ├── pipeline.py               # detect -> align -> crop -> predict -> JSON
│   ├── server.py                 # HTTP API: GET /, GET /api/health, POST /api/analyze
│   └── static/                   # Browser UI (vanilla JavaScript, no build step)
│       ├── index.html
│       ├── app.js
│       └── styles.css
├── scripts/
│   ├── download_checkpoints.sh   # Downloads the released checkpoints into checkpoints/ (resumable)
│   ├── run_app.sh                # Creates .venv, installs requirements-app.txt, starts the demo
│   └── e2e_fake_camera.mjs       # Playwright end-to-end check with a fake webcam
├── tests/                        # pytest suite for the demo (alignment, detector, models, API)
├── docs/
│   └── superpowers/specs/        # Design spec of the webcam demo
├── training/
│   ├── training_fahr_face/       # Training code for the foundation model
│   │   └── run.py                # Script to train the FAHR-Face foundation model
│   ├── train_fahr_faceage/       # Training code for the age estimation model
│   │   ├── run_pretraining.py    # Pretraining script for the age model
│   │   ├── run_finetuning.py     # Finetuning script for the age model
│   │   ├── loader.py             # Data loading utilities for age estimation
│   │   ├── optimizer.py          # Optimizer configuration
│   │   └── ViTForAgeEstimation.py # Age model architecture
│   └── train_fahr_facesurvival/  # Training code for the survival analysis model
│       ├── run.py                # Script to train the survival model
│       ├── concordance_index.py  # Evaluation metrics for survival analysis
│       ├── loader.py             # Data loading utilities for survival analysis
│       ├── optimizer.py          # Optimizer configuration
│       └── ViTForSurvivalAnalysis.py # Survival model architecture
└── other/
    └── face_crop_align.py        # Utilities for face cropping and alignment
```

## Model Architecture

### FAHR-Face (Foundation Model)

The foundation model is based on the Vision Transformer (ViT) architecture, specifically the Masked Autoencoder method. The model was trained on more than 40 million facial images from WebFace42M, the cleaned subset of the WebFace260M benchmark.

Key features:

- Transformer-based architecture with self-attention mechanisms
- Trained using the Masked Autoencoder method where a high portion (75%) of the image patches are randomly masked
- The encoder processes the visible patches, and a learnable mask token is added at the positions of the masked patches
- The decoder reconstructs the raw pixel values for the masked positions based on the encoded visible patches

### FAHR-FaceAge (Age Estimation Model)

The age estimation model builds upon the foundation model to predict chronological age from facial images:

- Uses transfer learning from the pre-trained FAHR-Face model
- Two-stage fine-tuning on 743,152 age-labeled photographs from ten public datasets: first eight oversampled datasets (IMDB-WIKI, KANFace, FG-NET, CACD, AFAD, MegaAge, MORPH, LAGENDA), then age-balanced fine-tuning on UTK and AgeDB
- Trained with L1 loss (Mean Absolute Error) to predict age as a continuous value

### FAHR-FaceSurvival (Survival Analysis Model)

The survival analysis model is designed to predict health risks from facial images:

- Built on top of the FAHR-Face foundation model
- Trained using a ranking loss function optimized for survival analysis
- Evaluated using the concordance index (C-index) metric, which is only used for validation and not for training loss

## Usage

### Face Preprocessing

**Detect, crop, and align one face per image before running FAHR-FaceAge.** Set `input_dir` and `output_dir` in [other/face_crop_align.py](other/face_crop_align.py), then run:

```bash
python other/face_crop_align.py
```

Use the resulting face crops as input. The age inference example converts them to 112 × 112 RGB pixels and applies the required normalization. Images that are already cropped and aligned can be used directly.

### Running Inference

For **standalone age inference with the released checkpoint**, use the [Hugging Face example](https://huggingface.co/fridolin22/FAHR-Face#fahr-faceage-inference). It requires only `fahr_faceage.pth`.

The existing `run.py` combines age and survival inference. It also requires `checkpoints/fahr_facesurvival.pth`, which is not distributed. If you already have access to that checkpoint:

1. First, download the model checkpoints as described in the [Model Checkpoints](#model-checkpoints) section.

2. Download the test images as described in the [Test Images](#test-images) section.

3. Set `perform_cropping_alignment=False` when using images that are already cropped and aligned, then run the main script:

```bash
python run.py
```

The script is pre-configured with the following paths:

- `age_checkpoint_path`: "checkpoints/fahr_faceage.pth" - Path to the age estimation model
- `survival_checkpoint_path`: "checkpoints/fahr_facesurvival.pth" - Path to the survival analysis model
- `photo_folder`: "synthetic_dataset_cropped_aligned" - Path to the folder with test photos

You can modify these paths in the script if needed for your specific setup.

### Training Models

Set the dataset and checkpoint paths in each script before training. Run each command from the indicated directory, starting at the repository root.

#### Foundation Model (FAHR-Face)

```bash
cd training/training_fahr_face
python run.py
```

#### Age Estimation Model (FAHR-FaceAge)

For pretraining:
```bash
cd training/train_fahr_faceage
python run_pretraining.py
```

For finetuning:
```bash
cd training/train_fahr_faceage
python run_finetuning.py
```

#### Survival Analysis Model (FAHR-FaceSurvival)

```bash
cd training/train_fahr_facesurvival
python run.py
```

## Live webcam demo (web UI)

A small local web app in [`app/`](app/) puts FAHR-FaceAge behind a webcam. Your browser streams frames to a server running on your own machine; the server detects the faces, preprocesses each one the way the paper did (detect → eye-based alignment → crop → 112 × 112 → normalization) and returns the **predicted face age**, which is drawn live over the video. **FAHR-FaceSurvival relative risk is shown only if you have `checkpoints/fahr_facesurvival.pth`**; otherwise the UI marks it as *not available*, because those weights are not publicly released. It is a research demo for a laptop: no TensorFlow, no GPU required, no build step, no external scripts.

### Quick start

1. Download the released age checkpoint into `checkpoints/` (resumable; add `--with-foundation` to also fetch `fahr_face.pth`):
```bash
bash scripts/download_checkpoints.sh
```

2. Create `.venv`, install `requirements-app.txt` and start the server:
```bash
bash scripts/run_app.sh
```

3. Open http://localhost:8000, read the consent note and allow camera access.

The manual equivalent, in any Python 3.10+ environment:

```bash
pip install -r requirements-app.txt
python -m app
```

`run_app.sh` installs the CPU-only PyTorch wheels on Linux (set `FAHR_GPU=1` for the default CUDA wheels), passes every `FAHR_*` variable through to the server and accepts `--skip-install` to start without touching pip. The server has no command-line flags; it is configured entirely through the environment variables listed under [Configuration](#configuration).

### What you see

- Boxes around the detected faces on the live video, optionally with the five landmarks, and a label such as `Age 42 · Risk n/a`.
- One card per face in the side panel with the smoothed age, the **relative risk score (uncalibrated)** or *not available* with the reason, the detection confidence and the 112 × 112 aligned crop ("what the model sees").
- Status chips for the detector, the device, the age model (ready / random weights / missing) and the survival model (ready / not available), plus banners for a missing checkpoint (with the download command), random weights, insecure context and camera errors.
- Three modes: **Continuous** (one request per frame; the next frame is sent when the previous answer arrives), **Snapshot** (freeze and analyze a single frame) and **Upload image** (analyze a photo, no camera needed). Mirror and landmark toggles.

### Face detection

The default detector is OpenCV's **YuNet** (`cv2.FaceDetectorYN`): a single ~230 KB ONNX model, downloaded once from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) into `~/.cache/fahr-face` (checksum verified), fast enough for live video on a CPU. The crop and alignment geometry applied to each detection is ported from `retina-face` 0.0.17, the package used by [`other/face_crop_align.py`](other/face_crop_align.py), so the age model sees the same kind of crop as during training. `FAHR_DETECTOR=retinaface` switches to the original RetinaFace package if it is installed (`pip install retina-face`; this pulls in TensorFlow). YuNet boxes can differ slightly from RetinaFace boxes, which may shift estimated ages by a few years; the active detector is always shown in the header.

### Configuration

All settings are environment variables. Relative paths are resolved from the directory you start the server in (the repository root when using `run_app.sh`).

| Variable | Default | Meaning |
|----------|---------|---------|
| `FAHR_AGE_CHECKPOINT` | `checkpoints/fahr_faceage.pth` | FAHR-FaceAge checkpoint. |
| `FAHR_SURVIVAL_CHECKPOINT` | `checkpoints/fahr_facesurvival.pth` | FAHR-FaceSurvival checkpoint (optional; not distributed). |
| `FAHR_DETECTOR` | `auto` | `auto` / `yunet` / `retinaface` (`auto` selects YuNet). |
| `FAHR_DETECTION_THRESHOLD` | detector default (YuNet: `0.7`) | Minimum detection score. |
| `FAHR_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `mps` (`auto` prefers CUDA, then MPS, then CPU). |
| `FAHR_ALLOW_RANDOM_WEIGHTS` | `0` | Development and tests only: random initialization when a checkpoint is missing. Output is meaningless and every response carries a warning. |
| `FAHR_MAX_FACES` | `5` | Faces analyzed per frame. |
| `FAHR_MAX_UPLOAD_BYTES` | `8388608` | Upload limit (8 MB). |
| `FAHR_MODEL_CACHE_DIR` | `~/.cache/fahr-face` | Where the YuNet model is cached. |
| `FAHR_YUNET_MODEL` | (unset) | Use this YuNet ONNX file instead of downloading one. |
| `FAHR_HOST` / `FAHR_PORT` | `127.0.0.1` / `8000` | Bind address and port. |

Example: `FAHR_PORT=8080 FAHR_DEVICE=cpu python -m app`

### Privacy and ethics

- Frames are sent only to the server you started, on your own machine (`127.0.0.1` by default). They are processed in memory and discarded; nothing about faces is written to disk or logged, and API responses carry `Cache-Control: no-store`. The page states this in a consent note before the camera starts.
- The demo detects and analyzes faces; it does not identify people, and it stores nothing.
- Like the rest of this repository, the demo is intended for academic research (see [Ethical Use](#ethical-use)). It is not a medical device and provides no medical advice. The survival output, when available, is an uncalibrated relative risk score from a research model, not a probability.

### Troubleshooting

- **Camera blocked or no permission prompt** – allow the camera for this site in the browser's permission settings (the camera icon in the address bar), then press *Start camera* again. Another application may also be holding the camera.
- **"Insecure context" banner** – browsers expose the camera only on `localhost` / `127.0.0.1` or over HTTPS. If the server runs on another machine (`FAHR_HOST=0.0.0.0`), reach it through an SSH tunnel (`ssh -L 8000:localhost:8000 <host>`) or put HTTPS in front of it. Upload mode works regardless.
- **Slow or choppy** – CPU inference takes a few hundred milliseconds per frame. The page already downsizes frames to at most 640 px wide and never sends the next frame before the previous answer arrives. Lower `FAHR_MAX_FACES`, or set `FAHR_DEVICE=cuda` / `mps` if you have a GPU.
- **"Random weights" banner** – a checkpoint is missing and the server was started with `FAHR_ALLOW_RANDOM_WEIGHTS=1`. The numbers are meaningless: download the checkpoint and restart without that variable.
- **Age model shown as missing** – no file at `FAHR_AGE_CHECKPOINT`; run `bash scripts/download_checkpoints.sh` and restart the server.
- **Survival shown as not available** – expected unless you have `checkpoints/fahr_facesurvival.pth`; the weights are not publicly released.
- **Detector error (HTTP 503)** – the YuNet model could not be downloaded. Fetch [`face_detection_yunet_2023mar.onnx`](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) manually into `~/.cache/fahr-face/`, or point `FAHR_YUNET_MODEL` at it.

### Tests

```bash
pip install pytest httpx   # test-only extras
python -m pytest -q
```

The suite covers the alignment port (compared with the vendored retina-face functions), the YuNet detector (public test photos are downloaded into `tests/fixtures/.cache/`; skipped offline), model loading with random weights and the HTTP API. One additional fidelity test runs only if `retina-face` is installed.

An end-to-end check drives the real page in Chromium with a fake webcam. Start the server first (with `FAHR_ALLOW_RANDOM_WEIGHTS=1` if you have no checkpoint), then run:

```bash
node scripts/e2e_fake_camera.mjs --video <file.y4m>
```

It needs Node 18+, [Playwright](https://playwright.dev) with Chromium (`npm install -g playwright && npx playwright install chromium`) and ffmpeg (only when `<file.y4m>` does not exist yet; it is then built from a test portrait). The script accepts the consent note, waits for a face card with an age, exercises Snapshot mode, saves a screenshot to `e2e-artifacts/webcam.png` and prints one JSON line.

## Dataset Information

FAHR-Face was pretrained on WebFace42M, the cleaned subset of WebFace260M, obtained from the original dataset authors under their data-use agreement. Researchers seeking access should contact the WebFace260M authors and follow their access procedures. The training images cover varied photographic conditions, poses, and facial appearances.

FAHR-FaceAge development used 743,152 photographs from ten public datasets. The first training stage used IMDB-WIKI, KANFace, FG-NET, CACD, AFAD, MegaAge, MORPH, and LAGENDA; the second stage used UTK and AgeDB. APPA-REAL (6,783 photographs) was used for evaluation only, with a reported mean absolute error of 5.1 years.

Clinical evaluation used the Harvard-RT cohort (38,183 patients) and independent Maastro-RT cohort (4,892 patients). Clinical photographs and patient-level data are not distributed in this repository.

## Environment Setup

The original training and face-preprocessing environment uses Python 3.8 and Miniforge (a free and open-source conda distribution that uses conda-forge as the default channel). For standalone age inference, the [model card](https://huggingface.co/fridolin22/FAHR-Face#setup) also provides a tested Python 3.11 setup. To create the original environment:

1. Install Miniforge if you don't have it already:
   - Download from: https://github.com/conda-forge/miniforge

2. Create and activate the environment:
```bash
conda env create -f facefoundation.yml
conda activate facefoundation
```

## Model Checkpoints

The released **FAHR-Face and FAHR-FaceAge** checkpoints are hosted together on [Hugging Face](https://huggingface.co/fridolin22/FAHR-Face). No login is required. Download them directly:

- [fahr_face.pth](https://huggingface.co/fridolin22/FAHR-Face/resolve/main/fahr_face.pth?download=true) — Foundation model
- [fahr_faceage.pth](https://huggingface.co/fridolin22/FAHR-Face/resolve/main/fahr_faceage.pth?download=true) — Age estimation model

Using the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli), download both into the `checkpoints` folder:

```bash
hf download fridolin22/FAHR-Face fahr_face.pth fahr_faceage.pth --local-dir checkpoints
```

These are original training checkpoints containing `model_state_dict` and optimizer state. Loading examples are available in the [model card](https://huggingface.co/fridolin22/FAHR-Face). FAHR-FaceAge inference needs only the age checkpoint. FAHR-FaceSurvival weights are not publicly released.

## Test Images

The synthetic test images are available for download:

[Download Synthetic Test Images](https://www.dropbox.com/scl/fo/hm9a5x60vya2810gyqty9/AD4yp_v6UY-1HLqMKy902X8?rlkey=d8hfle133ecsvh4bp7ripzeq8&st=q9g357io&dl=0)

After downloading, extract the files to create a `synthetic_dataset_cropped_aligned` directory in the root of the repository. This folder contains pre-processed face images used for testing the models.

Alternatively, you can run the pipeline with your own images:

- For raw images, set `perform_cropping_alignment=True` in `run.py` to detect/align faces on the fly.
- For preprocessed images, set `perform_cropping_alignment=False` and point `photo_folder` to your cropped/aligned images (e.g., `synthetic_dataset_cropped_aligned`).

## Performance

### Inference Time
The FAHR models are designed to be computationally efficient. Inference time for a single image is:
- Less than 5 seconds on a standard consumer-grade CPU
- Even faster on GPU-enabled systems

## Citation

Please cite: Haugg F, Lee G, He J, et al. **Foundation artificial intelligence models for health recognition by use of face photographs (FAHR-Face): a model development and validation study.** *The Lancet Digital Health*. 2026. https://doi.org/10.1016/j.landig.2026.101056

Fridolin Haugg and Grace Lee contributed equally. The [earlier preprint](https://arxiv.org/abs/2506.14909) is also available.

## Ethical Use

This software is intended for academic research. Any use for discrimination, surveillance, or harmful applications is prohibited. Users must follow appropriate research ethics and obtain necessary approvals.

## Licensing

This repository is multi-licensed:

- Code: Apache License 2.0.
- Model weights: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0).
- Associated journal article: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 (CC BY-NC-ND 4.0).

Full texts: https://www.apache.org/licenses/LICENSE-2.0 and https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
