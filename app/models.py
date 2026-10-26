"""FAHR-FaceAge / FAHR-FaceSurvival models: loading and batched prediction.

Architectures and preprocessing are exactly those of ``run.py``:

* age:      ``ViTForAgeRegression(ViTConfig(image_size=112, num_hidden_layers=12, num_attention_heads=12))``
* survival: ``ViTForSurvivalAnalysis(ViTConfig(num_labels=1, image_size=112, num_hidden_layers=12,
  num_attention_heads=12))``
* transform: ``Resize((112, 112)) -> ToTensor() -> Normalize(ImageNet mean/std)`` on PIL RGB images.

A missing or broken checkpoint never raises here; the model state records what
happened so the server can start and report it.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTConfig

import app  # noqa: F401  (puts the repository root on sys.path)
from app.config import Settings
from ViTForAgeEstimation import ViTForAgeRegression
from ViTForSurvivalAnalysis import ViTForSurvivalAnalysis

log = logging.getLogger("fahr.models")

SURVIVAL_MISSING_REASON = "Checkpoint not found. FAHR-FaceSurvival weights are not publicly released."
RANDOM_WEIGHTS_WARNING = (
    "Age model is running with RANDOM weights (FAHR_ALLOW_RANDOM_WEIGHTS=1): predicted ages are "
    "meaningless. Download fahr_faceage.pth into checkpoints/ for real predictions."
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 112

WEIGHTS_CHECKPOINT = "checkpoint"
WEIGHTS_RANDOM = "random"
WEIGHTS_NONE = "none"


@dataclass(frozen=True)
class Prediction:
    """Per-face model outputs; ``None`` when the corresponding model is unavailable."""

    age: Optional[float] = None
    risk_score: Optional[float] = None


@dataclass
class ModelState:
    """What happened when loading one model."""

    name: str
    checkpoint: str
    weights: str = WEIGHTS_NONE  # checkpoint | random | none
    error: Optional[str] = None  # human readable reason when weights != checkpoint

    @property
    def available(self) -> bool:
        return self.weights in (WEIGHTS_CHECKPOINT, WEIGHTS_RANDOM)

    def health_dict(self) -> dict:
        return {
            "available": self.available,
            "weights": self.weights,
            "checkpoint": self.checkpoint,
            "error": self.error,
        }

    def summary_dict(self) -> dict:
        return {"available": self.available, "weights": self.weights, "reason": self.error}


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #


def resolve_device(name: str = "auto") -> str:
    """``auto`` -> cuda if available, else mps if available, else cpu."""
    key = (name or "auto").strip().lower()
    if key == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    try:
        torch.device(key)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"Unknown device {name!r}; use auto, cpu, cuda or mps") from exc
    if key.startswith("cuda") and not torch.cuda.is_available():
        log.warning("FAHR_DEVICE=%s requested but CUDA is not available; falling back to cpu", name)
        return "cpu"
    if key == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            log.warning("FAHR_DEVICE=mps requested but MPS is not available; falling back to cpu")
            return "cpu"
    return key


# --------------------------------------------------------------------------- #
# Checkpoint loading
# --------------------------------------------------------------------------- #

# transformers < 5 named the ViT encoder weights differently from transformers >= 5.
# The released FAHR checkpoints were saved with the old names; map in both directions
# so a checkpoint loads regardless of the installed transformers version.
_LEGACY_TO_CURRENT: Tuple[Tuple[str, str], ...] = (
    (r"^vit\.encoder\.layer\.(\d+)\.attention\.attention\.query\.", r"vit.layers.\1.attention.q_proj."),
    (r"^vit\.encoder\.layer\.(\d+)\.attention\.attention\.key\.", r"vit.layers.\1.attention.k_proj."),
    (r"^vit\.encoder\.layer\.(\d+)\.attention\.attention\.value\.", r"vit.layers.\1.attention.v_proj."),
    (r"^vit\.encoder\.layer\.(\d+)\.attention\.output\.dense\.", r"vit.layers.\1.attention.o_proj."),
    (r"^vit\.encoder\.layer\.(\d+)\.intermediate\.dense\.", r"vit.layers.\1.mlp.fc1."),
    (r"^vit\.encoder\.layer\.(\d+)\.output\.dense\.", r"vit.layers.\1.mlp.fc2."),
    (r"^vit\.encoder\.layer\.(\d+)\.", r"vit.layers.\1."),
)
_CURRENT_TO_LEGACY: Tuple[Tuple[str, str], ...] = (
    (r"^vit\.layers\.(\d+)\.attention\.q_proj\.", r"vit.encoder.layer.\1.attention.attention.query."),
    (r"^vit\.layers\.(\d+)\.attention\.k_proj\.", r"vit.encoder.layer.\1.attention.attention.key."),
    (r"^vit\.layers\.(\d+)\.attention\.v_proj\.", r"vit.encoder.layer.\1.attention.attention.value."),
    (r"^vit\.layers\.(\d+)\.attention\.o_proj\.", r"vit.encoder.layer.\1.attention.output.dense."),
    (r"^vit\.layers\.(\d+)\.mlp\.fc1\.", r"vit.encoder.layer.\1.intermediate.dense."),
    (r"^vit\.layers\.(\d+)\.mlp\.fc2\.", r"vit.encoder.layer.\1.output.dense."),
    (r"^vit\.layers\.(\d+)\.", r"vit.encoder.layer.\1."),
)


def _apply_rules(rules: Iterable[Tuple[str, str]], key: str) -> str:
    for pattern, repl in rules:
        new_key, n = re.subn(pattern, repl, key)
        if n:
            return new_key
    return key


def remap_state_dict_keys(state_dict: Dict[str, torch.Tensor], expected_keys: Iterable[str]) -> Dict[str, torch.Tensor]:
    """Rename checkpoint keys to the names the instantiated model expects.

    Handles the transformers 4.x <-> 5.x ViT renames and a ``module.`` prefix
    left by ``DataParallel``.  When no mapping produces the expected key set the
    input is returned unchanged so ``load_state_dict`` reports the mismatch.
    """
    expected: Set[str] = set(expected_keys)
    candidates: List[Dict[str, torch.Tensor]] = [state_dict]
    if state_dict and all(k.startswith("module.") for k in state_dict):
        candidates.append({k[len("module.") :]: v for k, v in state_dict.items()})
    for base in list(candidates):
        if set(base) == expected:
            return base
        for rules in (_LEGACY_TO_CURRENT, _CURRENT_TO_LEGACY):
            renamed = {_apply_rules(rules, k): v for k, v in base.items()}
            if set(renamed) == expected:
                return renamed
    return state_dict


def load_checkpoint_state_dict(path: str, device: str) -> Dict[str, torch.Tensor]:
    """``torch.load`` a checkpoint and return its ``model_state_dict``.

    Tries ``weights_only=True`` first and falls back to the unsafe loader with a
    logged warning (older checkpoints may pickle non-tensor objects).
    """
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:  # noqa: BLE001 - we deliberately fall back
        log.warning(
            "torch.load(weights_only=True) failed for %s (%s: %s); retrying with weights_only=False. "
            "Only load checkpoints you trust.",
            path,
            type(exc).__name__,
            exc,
        )
        checkpoint = torch.load(path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        state_dict = checkpoint  # a bare state dict
    else:
        raise ValueError("checkpoint has no 'model_state_dict' entry")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("'model_state_dict' is empty or not a mapping")
    return state_dict


def build_age_model() -> ViTForAgeRegression:
    config = ViTConfig(image_size=112, num_hidden_layers=12, num_attention_heads=12)
    return ViTForAgeRegression(config)


def build_survival_model() -> ViTForSurvivalAnalysis:
    config = ViTConfig(num_labels=1, image_size=112, num_hidden_layers=12, num_attention_heads=12)
    return ViTForSurvivalAnalysis(config)


# --------------------------------------------------------------------------- #
# FaceModels
# --------------------------------------------------------------------------- #


class FaceModels:
    """Holds the age and survival models and runs batched inference on RGB crops."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = resolve_device(settings.device)
        self._device = torch.device(self.device)
        self._lock = threading.Lock()
        self._transform = transforms.Compose(
            [
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        self.age_model, self.age = self._load(
            name="age",
            checkpoint=settings.age_checkpoint,
            build=build_age_model,
            allow_random=settings.allow_random_weights,
            missing_reason=(
                f"Checkpoint not found at {settings.age_checkpoint}. Download fahr_faceage.pth from the "
                "FAHR-Face Hugging Face repository into checkpoints/ (see README)."
            ),
        )
        # The survival weights are not public; random weights would invite misreading a
        # meaningless "risk", so this model never falls back to random initialisation.
        self.survival_model, self.survival = self._load(
            name="survival",
            checkpoint=settings.survival_checkpoint,
            build=build_survival_model,
            allow_random=False,
            missing_reason=SURVIVAL_MISSING_REASON,
        )

    # ----------------------------------------------------------------- loading

    def _load(
        self,
        *,
        name: str,
        checkpoint: str,
        build: Callable[[], torch.nn.Module],
        allow_random: bool,
        missing_reason: str,
    ) -> Tuple[Optional[torch.nn.Module], ModelState]:
        state = ModelState(name=name, checkpoint=checkpoint)
        path = os.path.expanduser(checkpoint) if checkpoint else ""

        if not path or not os.path.isfile(path):
            if allow_random:
                try:
                    model = build().to(self._device).eval()
                except Exception as exc:  # noqa: BLE001
                    log.exception("Could not build the %s model", name)
                    state.error = f"{missing_reason} Building the model failed: {type(exc).__name__}: {exc}"
                    return None, state
                state.weights = WEIGHTS_RANDOM
                state.error = (
                    f"{missing_reason} Using randomly initialised weights because "
                    "FAHR_ALLOW_RANDOM_WEIGHTS=1; outputs are meaningless."
                )
                log.warning("%s model: %s", name, state.error)
                return model, state
            state.error = missing_reason
            log.warning("%s model unavailable: %s", name, missing_reason)
            return None, state

        try:
            state_dict = load_checkpoint_state_dict(path, self.device)
            model = build()
            state_dict = remap_state_dict_keys(state_dict, model.state_dict().keys())
            model.load_state_dict(state_dict, strict=True)
            model = model.to(self._device).eval()
        except Exception as exc:  # noqa: BLE001 - must never crash startup
            log.exception("Failed to load the %s checkpoint %s", name, checkpoint)
            message = str(exc).strip().splitlines()
            detail = message[0] if message else ""
            if len(detail) > 300:
                detail = detail[:300] + "..."
            state.error = f"Failed to load checkpoint {checkpoint}: {type(exc).__name__}: {detail}"
            return None, state

        state.weights = WEIGHTS_CHECKPOINT
        state.error = None
        log.info("%s model loaded from %s on %s", name, checkpoint, self.device)
        return model, state

    # --------------------------------------------------------------- inference

    @staticmethod
    def _to_pil(crop: np.ndarray) -> Image.Image:
        arr = np.asarray(crop)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError("crops must be H x W x 3 RGB arrays")
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(np.ascontiguousarray(arr), mode="RGB")

    def predict(self, crops_rgb: Sequence[np.ndarray]) -> List[Prediction]:
        """Run both models on a batch of RGB uint8 crops (any size; resized to 112x112)."""
        n = len(crops_rgb)
        if n == 0:
            return []
        if self.age_model is None and self.survival_model is None:
            return [Prediction() for _ in range(n)]

        batch = torch.stack([self._transform(self._to_pil(c)) for c in crops_rgb])

        ages: Optional[List[float]] = None
        risks: Optional[List[float]] = None
        with self._lock, torch.inference_mode():
            batch = batch.to(self._device, non_blocking=False)
            # return_dict=True is passed explicitly: the repository classes otherwise
            # consult config.use_return_dict, which transformers >= 5 deprecates.
            if self.age_model is not None:
                out = self.age_model(pixel_values=batch, return_dict=True)
                # The repo model calls .squeeze(): batch size 1 gives a 0-d tensor.
                ages = out.age_output.reshape(-1).detach().float().cpu().tolist()
            if self.survival_model is not None:
                out = self.survival_model(pixel_values=batch, return_dict=True)
                risks = out.logits.reshape(-1).detach().float().cpu().tolist()

        if ages is not None and len(ages) != n:
            raise RuntimeError(f"age model returned {len(ages)} values for {n} crops")
        if risks is not None and len(risks) != n:
            raise RuntimeError(f"survival model returned {len(risks)} values for {n} crops")

        return [
            Prediction(
                age=float(ages[i]) if ages is not None else None,
                risk_score=float(risks[i]) if risks is not None else None,
            )
            for i in range(n)
        ]

    # ----------------------------------------------------------------- reports

    def health(self) -> Dict[str, dict]:
        return {"age": self.age.health_dict(), "survival": self.survival.health_dict()}

    def summary(self) -> Dict[str, dict]:
        return {"age": self.age.summary_dict(), "survival": self.survival.summary_dict()}

    def warnings(self) -> List[str]:
        out: List[str] = []
        if self.age.weights == WEIGHTS_RANDOM:
            out.append(RANDOM_WEIGHTS_WARNING)
        elif not self.age.available:
            out.append(f"Age model unavailable: {self.age.error}")
        return out


__all__ = [
    "FaceModels",
    "ModelState",
    "Prediction",
    "SURVIVAL_MISSING_REASON",
    "RANDOM_WEIGHTS_WARNING",
    "build_age_model",
    "build_survival_model",
    "load_checkpoint_state_dict",
    "remap_state_dict_keys",
    "resolve_device",
]
