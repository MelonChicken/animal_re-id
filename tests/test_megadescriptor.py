"""Integration tests for the real MegaDescriptor checkpoint.

These download/load a multi-hundred-megabyte checkpoint, so they are opt-in:

    $env:RUN_MODEL_TESTS = "1"; uv run pytest -m integration

The small ``MegaDescriptor-T-224`` variant is used to keep the test cheap; the
production config uses ``MegaDescriptor-L-384``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from animal_re_id.models.megadescriptor import (
    FeatureExtractionError,
    build_transform,
    describe_model,
    extract_features,
    load_model,
    validate_features,
)

pytestmark = pytest.mark.integration

TEST_MODEL = "hf-hub:BVRA/MegaDescriptor-T-224"


@pytest.fixture(scope="module")
def model():
    if os.environ.get("RUN_MODEL_TESTS") != "1":
        pytest.skip("Set RUN_MODEL_TESTS=1 to run checkpoint integration tests")
    device = "cuda" if _cuda_available() else "cpu"
    return load_model(TEST_MODEL, device=device), device


def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


@pytest.fixture
def image_frame(synthetic_root):
    from animal_re_id.data.wildlife import load_polar_bear_vid_id
    from animal_re_id.data.splits import add_sequence_columns, representative_frames

    catalogue = add_sequence_columns(load_polar_bear_vid_id(synthetic_root))
    return representative_frames(catalogue).head(6).reset_index(drop=True)


def test_checkpoint_is_frozen_and_described(model) -> None:
    network, _ = model
    assert not network.training
    assert all(not parameter.requires_grad for parameter in network.parameters())

    description = describe_model(TEST_MODEL, network)
    # timm reports the Hub repo id without the "hf-hub:" prefix.
    assert description["hf_hub_id"].endswith(TEST_MODEL.split(":", 1)[-1])
    assert description["parameters"] > 1_000_000
    assert description["architecture"]


def test_transform_matches_the_checkpoint(model) -> None:
    network, _ = model
    transform = build_transform(network)
    assert isinstance(transform, object)
    # A 256x128 PolarBearVidID frame must come out square at the model size.
    from PIL import Image

    tensor = transform(Image.new("RGB", (256, 128), color=(10, 20, 30)))
    assert tensor.shape[0] == 3 and tensor.shape[1] == tensor.shape[2]


def test_embeddings_have_expected_shape_and_are_finite(model, image_frame: pd.DataFrame) -> None:
    network, device = model
    features = extract_features(
        network, build_transform(network), image_frame, batch_size=4, device=device
    )
    assert features.ndim == 2
    assert features.shape[0] == len(image_frame)
    assert features.shape[1] > 0
    assert np.isfinite(features).all()
    validate_features(features, expected_rows=len(image_frame))


def test_same_image_gives_the_same_embedding(model, image_frame: pd.DataFrame) -> None:
    network, device = model
    transform = build_transform(network)
    single = image_frame.iloc[[0]]

    first = extract_features(network, transform, single, batch_size=1, device=device)
    second = extract_features(network, transform, single, batch_size=1, device=device)
    assert np.allclose(first, second, atol=1e-5)


def test_validate_features_rejects_non_finite() -> None:
    features = np.ones((2, 4), dtype="float32")
    features[1, 2] = np.nan
    with pytest.raises(FeatureExtractionError, match="non-finite"):
        validate_features(features, expected_rows=2)


def test_validate_features_rejects_wrong_row_count() -> None:
    with pytest.raises(FeatureExtractionError, match="embeddings for"):
        validate_features(np.ones((3, 4), dtype="float32"), expected_rows=2)


def test_validate_features_rejects_all_zero() -> None:
    with pytest.raises(FeatureExtractionError, match="all zeros"):
        validate_features(np.zeros((2, 4), dtype="float32"), expected_rows=2)
