"""Age / survival model loading and batched prediction."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.config import Settings  # noqa: E402
from app.models import (  # noqa: E402
    SURVIVAL_MISSING_REASON,
    FaceModels,
    Prediction,
    remap_state_dict_keys,
    resolve_device,
)


def _crops(n: int, seed: int = 0, size=(130, 110)) -> list:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, size=(size[0] + i, size[1], 3), dtype=np.uint8) for i in range(n)]


def test_random_weights_mode_gives_finite_floats(face_models):
    assert face_models.age.weights == "random"
    assert face_models.age.available is True
    assert face_models.age.error  # human readable reason for the random weights
    assert face_models.survival.weights == "none"
    assert face_models.survival.available is False
    assert face_models.survival.error == SURVIVAL_MISSING_REASON

    preds = face_models.predict(_crops(1))
    assert len(preds) == 1
    assert isinstance(preds[0], Prediction)
    assert isinstance(preds[0].age, float) and math.isfinite(preds[0].age)
    assert preds[0].risk_score is None


def test_batch_sizes_one_and_n_are_consistent(face_models):
    crops = _crops(3, seed=1)
    single = face_models.predict(crops[:1])
    batch = face_models.predict(crops)
    assert len(single) == 1 and len(batch) == 3
    assert all(isinstance(p.age, float) and math.isfinite(p.age) for p in batch)
    assert batch[0].age == pytest.approx(single[0].age, abs=1e-3)
    # Different inputs should (almost surely) give different outputs.
    assert len({round(p.age, 4) for p in batch}) > 1


def test_empty_input(face_models):
    assert face_models.predict([]) == []


def test_missing_checkpoint_without_random_weights_is_none_and_never_raises():
    settings = Settings(
        allow_random_weights=False,
        age_checkpoint="/nonexistent/age.pth",
        survival_checkpoint="/nonexistent/surv.pth",
        device="cpu",
    )
    models = FaceModels(settings)
    assert models.age.weights == "none"
    assert models.age.available is False
    assert "/nonexistent/age.pth" in models.age.error
    assert models.survival.weights == "none"
    assert models.survival.error == SURVIVAL_MISSING_REASON
    preds = models.predict(_crops(2))
    assert preds == [Prediction(age=None, risk_score=None)] * 2
    health = models.health()
    assert health["age"]["available"] is False
    assert health["age"]["weights"] == "none"
    assert health["age"]["checkpoint"] == "/nonexistent/age.pth"
    assert health["survival"]["error"] == SURVIVAL_MISSING_REASON


def test_corrupt_checkpoint_is_reported_not_raised(tmp_path):
    bad = tmp_path / "age.pth"
    bad.write_bytes(b"\x00\x01 definitely not a torch checkpoint")
    settings = Settings(allow_random_weights=True, age_checkpoint=str(bad), device="cpu",
                        survival_checkpoint="/nonexistent/surv.pth")
    models = FaceModels(settings)
    # An existing but unreadable checkpoint is a user error: no silent random fallback.
    assert models.age.weights == "none"
    assert models.age.available is False
    assert models.age.error and str(bad) in models.age.error


def _legacy_keys(state_dict: dict) -> dict:
    """Rename transformers>=5 ViT keys to the transformers 4.x names used by the released weights."""
    import re

    out = {}
    for k, v in state_dict.items():
        k2 = re.sub(r"^vit\.layers\.(\d+)\.attention\.q_proj\.", r"vit.encoder.layer.\1.attention.attention.query.", k)
        k2 = re.sub(r"^vit\.layers\.(\d+)\.attention\.k_proj\.", r"vit.encoder.layer.\1.attention.attention.key.", k2)
        k2 = re.sub(r"^vit\.layers\.(\d+)\.attention\.v_proj\.", r"vit.encoder.layer.\1.attention.attention.value.", k2)
        k2 = re.sub(r"^vit\.layers\.(\d+)\.attention\.o_proj\.", r"vit.encoder.layer.\1.attention.output.dense.", k2)
        k2 = re.sub(r"^vit\.layers\.(\d+)\.mlp\.fc1\.", r"vit.encoder.layer.\1.intermediate.dense.", k2)
        k2 = re.sub(r"^vit\.layers\.(\d+)\.mlp\.fc2\.", r"vit.encoder.layer.\1.output.dense.", k2)
        k2 = re.sub(r"^vit\.layers\.(\d+)\.", r"vit.encoder.layer.\1.", k2)
        out[k2] = v
    return out


def test_checkpoint_roundtrip_including_legacy_key_names(tmp_path, face_models):
    """A checkpoint saved by run.py-era code (transformers 4.x key names) must load."""
    age_sd = face_models.age_model.state_dict()
    legacy = _legacy_keys(age_sd)
    if any(k.startswith("vit.layers.") for k in age_sd):
        # transformers >= 5: the legacy names differ and must be remapped on load
        assert not any(k.startswith("vit.layers.") for k in legacy)
        remapped = remap_state_dict_keys(legacy, set(age_sd))
        assert set(remapped) == set(age_sd)
    age_path = tmp_path / "fahr_faceage.pth"
    torch.save({"model_state_dict": legacy, "epoch": 1}, age_path)

    # Survival: a fresh random model saved in the native key format.
    from ViTForSurvivalAnalysis import ViTForSurvivalAnalysis
    from transformers import ViTConfig

    surv = ViTForSurvivalAnalysis(
        ViTConfig(num_labels=1, image_size=112, num_hidden_layers=12, num_attention_heads=12)
    )
    surv_path = tmp_path / "fahr_facesurvival.pth"
    torch.save({"model_state_dict": surv.state_dict()}, surv_path)

    settings = Settings(
        allow_random_weights=False,
        age_checkpoint=str(age_path),
        survival_checkpoint=str(surv_path),
        device="cpu",
    )
    loaded = FaceModels(settings)
    assert loaded.age.weights == "checkpoint"
    assert loaded.age.available is True
    assert loaded.age.error is None
    assert loaded.survival.weights == "checkpoint"
    assert loaded.survival.available is True

    crops = _crops(2, seed=5)
    expected = face_models.predict(crops)
    got = loaded.predict(crops)
    for e, g in zip(expected, got):
        assert g.age == pytest.approx(e.age, abs=1e-4)
        assert isinstance(g.risk_score, float) and math.isfinite(g.risk_score)

    # The survival head alone must agree with a direct forward pass.
    from PIL import Image
    from torchvision import transforms

    tf = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    surv.eval()
    with torch.inference_mode():
        ref = surv(tf(Image.fromarray(crops[0])).unsqueeze(0)).logits.reshape(-1)[0].item()
    assert got[0].risk_score == pytest.approx(ref, abs=1e-4)


def test_checkpoint_with_wrong_shapes_is_reported(tmp_path):
    bad_sd = {"age_head.weight": torch.zeros(1, 7), "age_head.bias": torch.zeros(1)}
    path = tmp_path / "age.pth"
    torch.save({"model_state_dict": bad_sd}, path)
    models = FaceModels(Settings(age_checkpoint=str(path), survival_checkpoint="/nonexistent/s.pth",
                                 device="cpu"))
    assert models.age.weights == "none"
    assert models.age.available is False
    assert models.age.error


def test_resolve_device():
    assert resolve_device("cpu") == "cpu"
    auto = resolve_device("auto")
    assert auto in {"cpu", "cuda", "mps"}
    with pytest.raises(ValueError):
        resolve_device("tpu-9000")


def test_summary_blocks(face_models):
    health = face_models.health()
    assert set(health) == {"age", "survival"}
    for block in health.values():
        assert set(block) == {"available", "weights", "checkpoint", "error"}
    summary = face_models.summary()
    for block in summary.values():
        assert set(block) == {"available", "weights", "reason"}
    assert summary["age"]["weights"] == "random"
    assert summary["survival"]["reason"] == SURVIVAL_MISSING_REASON
    assert face_models.warnings()  # random weights warning
    assert any("random" in w.lower() for w in face_models.warnings())
