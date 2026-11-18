"""Integration tests against the real DINOv2-S checkpoint.

These download the checkpoint (``facebook/dinov2-small``, ~90 MB), so they are
opt-in:

    $env:RUN_MODEL_TESTS = "1"; uv run pytest -m integration

What they pin is the contract the synthetic tests cannot check: the real token
layout (1 CLS + 256 patches for 224x224 / patch 14), the 384-dimensional output,
and that the three reductions really differ on a real image.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from animal_re_id.models.dinov2 import (
    DEFAULT_MODEL_NAME,
    SpatialPooler,
    build_transform,
    describe_model,
    extract_frame_features,
    load_model,
    spatial_pool,
)

pytestmark = pytest.mark.integration


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def model():
    if os.environ.get("RUN_MODEL_TESTS") != "1":
        pytest.skip("Set RUN_MODEL_TESTS=1 to run checkpoint integration tests")
    return load_model(DEFAULT_MODEL_NAME, device=_device())


@pytest.fixture
def frames(catalogue):
    frames = catalogue.head(2).copy()
    frames["path"] = frames["path"].astype(str)
    return frames.reset_index(drop=True)


def test_description_records_checkpoint_and_shape(model):
    description = describe_model(DEFAULT_MODEL_NAME, model)
    assert description["hidden_size"] == 384
    assert description["patch_size"] == 14
    assert description["frozen"] is True
    assert description["checkpoint_revision"], "checkpoint revision must be recorded"


def test_forward_pass_exposes_257_tokens(model):
    pixel_values = torch.zeros(1, 3, 224, 224, device=_device())
    with torch.no_grad():
        tokens = model(pixel_values=pixel_values).last_hidden_state
    assert tokens.shape == (1, 257, 384)


def test_reductions_differ_on_a_real_image(model, frames):
    transform = build_transform(224, "imagenet")
    features = extract_frame_features(
        model, transform, frames, reductions=("patches", "all", "cls"), batch_size=2, device=_device()
    )
    assert features.shape == (2, 3, 384)
    assert np.isfinite(features).all()
    patches, everything, cls = features[:, 0], features[:, 1], features[:, 2]
    assert not np.allclose(patches, cls)
    assert not np.allclose(everything, cls)


def test_extraction_is_deterministic(model, frames):
    transform = build_transform(224, "imagenet")
    first = extract_frame_features(model, transform, frames, batch_size=2, device=_device())
    second = extract_frame_features(model, transform, frames, batch_size=2, device=_device())
    assert np.allclose(first, second)


def test_pooler_matches_spatial_pool(model, frames):
    from PIL import Image

    transform = build_transform(224, "imagenet")
    image = transform(Image.new("RGB", (224, 224), (10, 20, 30))).unsqueeze(0).to(_device())
    with torch.no_grad():
        batched = model(pixel_values=image).last_hidden_state
    pooler = SpatialPooler(model, ("patches",)).to(_device())
    with torch.no_grad():
        pooled = pooler(image)
    assert torch.allclose(pooled[0], spatial_pool(batched, "patches")[0], atol=1e-5)
