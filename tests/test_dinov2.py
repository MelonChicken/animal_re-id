"""DINOv2 token reduction and preprocessing.

The three spatial reductions are the crux of the module: ``patches`` excludes
the CLS token, ``all`` includes it, and ``cls`` keeps only it. These tests use
hand-built token tensors, so they run without downloading a checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from animal_re_id.models.dinov2 import (
    DEFAULT_IMAGE_SIZE,
    FeatureExtractionError,
    build_transform,
    describe_model,
    l2_normalize,
    spatial_pool,
    validate_frame_features,
)


def make_tokens(batch: int = 2, patches: int = 4, hidden: int = 3) -> torch.Tensor:
    """CLS = 100, patches = 1..patches, so each reduction is identifiable."""
    tokens = torch.zeros(batch, 1 + patches, hidden)
    tokens[:, 0, :] = 100.0
    for index in range(patches):
        tokens[:, 1 + index, :] = float(index + 1)
    return tokens


# --------------------------------------------------------------------------- #
# Spatial pooling
# --------------------------------------------------------------------------- #
def test_patch_pool_excludes_cls():
    result = spatial_pool(make_tokens(patches=4), "patches")
    assert result.shape == (2, 3)
    assert torch.allclose(result, torch.full((2, 3), 2.5))


def test_all_pool_includes_cls():
    result = spatial_pool(make_tokens(patches=4), "all")
    assert torch.allclose(result, torch.full((2, 3), (100.0 + 1 + 2 + 3 + 4) / 5))


def test_cls_pool_keeps_only_the_first_token():
    result = spatial_pool(make_tokens(patches=4), "cls")
    assert torch.allclose(result, torch.full((2, 3), 100.0))


def test_reductions_are_actually_different():
    tokens = make_tokens()
    patches = spatial_pool(tokens, "patches")
    everything = spatial_pool(tokens, "all")
    cls = spatial_pool(tokens, "cls")
    assert not torch.allclose(patches, everything)
    assert not torch.allclose(patches, cls)


def test_pool_rejects_unknown_reduction_and_bad_shape():
    with pytest.raises(FeatureExtractionError):
        spatial_pool(make_tokens(), "median")
    with pytest.raises(FeatureExtractionError):
        spatial_pool(torch.zeros(2, 3), "patches")
    with pytest.raises(FeatureExtractionError):
        spatial_pool(torch.zeros(2, 1, 3), "patches")


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def test_imagenet_transform_normalizes_and_resizes():
    from PIL import Image

    transform = build_transform(224, "imagenet")
    image = Image.fromarray(np.full((128, 256, 3), 128, dtype=np.uint8))
    tensor = transform(image)
    assert tensor.shape == (3, 224, 224)
    # Mid-grey maps to a small residual, not to 0.5, once mean/std are applied.
    assert not torch.allclose(tensor, torch.full_like(tensor, 0.5), atol=1e-2)


def test_rovf_transform_is_divide_by_255_only():
    from PIL import Image

    transform = build_transform(DEFAULT_IMAGE_SIZE, "rovf")
    image = Image.fromarray(np.full((128, 256, 3), 255, dtype=np.uint8))
    tensor = transform(image)
    assert tensor.shape == (3, 224, 224)
    assert torch.allclose(tensor, torch.ones_like(tensor), atol=1e-5)


def test_transform_rejects_unknown_mode():
    with pytest.raises(FeatureExtractionError):
        build_transform(224, "clip")


# --------------------------------------------------------------------------- #
# Validation and embedding hygiene
# --------------------------------------------------------------------------- #
def test_validate_accepts_well_formed_features():
    features = np.ones((4, 2, 8), dtype="float32")
    validate_frame_features(features, expected_rows=4, expected_dim=8, expected_reductions=2)


@pytest.mark.parametrize(
    "features",
    [
        np.zeros((4, 2, 8)),
        np.full((4, 2, 8), np.nan),
        np.full((4, 2, 8), np.inf),
        np.ones((4, 8)),
    ],
)
def test_validate_rejects_bad_features(features):
    with pytest.raises(FeatureExtractionError):
        validate_frame_features(features)


def test_validate_rejects_shape_mismatches():
    features = np.ones((4, 2, 8), dtype="float32")
    with pytest.raises(FeatureExtractionError):
        validate_frame_features(features, expected_rows=5)
    with pytest.raises(FeatureExtractionError):
        validate_frame_features(features, expected_dim=7)
    with pytest.raises(FeatureExtractionError):
        validate_frame_features(features, expected_reductions=3)


def test_l2_normalize_produces_unit_vectors():
    normalized = l2_normalize(np.array([[3.0, 4.0], [0.0, 2.0]], dtype="float32"))
    assert np.allclose(np.linalg.norm(normalized, axis=1), [1.0, 1.0])


def test_l2_normalize_rejects_zero_norm():
    with pytest.raises(FeatureExtractionError):
        l2_normalize(np.zeros((2, 3), dtype="float32"))


# --------------------------------------------------------------------------- #
# Description
# --------------------------------------------------------------------------- #
def test_describe_model_records_checkpoint_without_a_model():
    description = describe_model("facebook/dinov2-small")
    assert description["name"] == "facebook/dinov2-small"
    assert description["checkpoint"] == "facebook/dinov2-small"
    assert description["library"] == "transformers"
