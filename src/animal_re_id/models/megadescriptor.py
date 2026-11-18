"""Frozen MegaDescriptor feature extractor.

MegaDescriptor checkpoints are distributed on the Hugging Face Hub and consumed
through ``timm``. Extraction itself is delegated to
``wildlife_tools.features.DeepFeatures`` so that batching, device handling and
the dataset interface are the ones the WildlifeTools authors maintain.

Pre-training caveat (recorded, not resolved): MegaDescriptor-L-384 is trained
on all publicly available ``wildlife-datasets`` re-identification datasets,
which include PolarBearVidID. Results on PolarBearVidID are therefore not
zero-shot and must not be compared with held-out benchmarks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms as T

from wildlife_tools.data import WildlifeDataset
from wildlife_tools.features import DeepFeatures

DEFAULT_MODEL_NAME = "hf-hub:BVRA/MegaDescriptor-L-384"


class FeatureExtractionError(RuntimeError):
    """Raised when extracted features are structurally invalid."""


def load_model(name: str = DEFAULT_MODEL_NAME, device: str = "cpu") -> torch.nn.Module:
    """Load a frozen, evaluation-mode MegaDescriptor backbone."""
    model = timm.create_model(name, pretrained=True, num_classes=0)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device)


def build_transform(model: torch.nn.Module) -> T.Compose:
    """Build the preprocessing recipe shipped with the checkpoint.

    MegaDescriptor's model card resizes straight to the square input size
    without cropping (PolarBearVidID frames are 256x128), so the transform is
    resize-and-normalize rather than a random-resized crop.
    """
    config = timm.data.resolve_model_data_config(model)
    height, width = config["input_size"][-2:]
    if height != width:
        raise FeatureExtractionError(f"Expected a square input size, got {config['input_size']}")
    mean = config.get("mean", (0.5, 0.5, 0.5))
    std = config.get("std", (0.5, 0.5, 0.5))
    return T.Compose(
        [
            T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def extract_features(
    model: torch.nn.Module,
    transform: T.Compose,
    metadata: pd.DataFrame,
    batch_size: int = 32,
    device: str = "cpu",
    num_workers: int = 0,
) -> np.ndarray:
    """Embed every row of ``metadata`` and validate the result."""
    dataset = WildlifeDataset(
        metadata.reset_index(drop=True),
        root=None,
        transform=transform,
        img_load="full",
        load_label=True,
    )
    extractor = DeepFeatures(
        model=model,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    features = np.asarray(extractor(dataset), dtype="float32")
    validate_features(features, expected_rows=len(metadata))
    return features


def validate_features(features: np.ndarray, expected_rows: int | None = None) -> None:
    """Reject non-finite or wrongly shaped embedding matrices."""
    if features.ndim != 2:
        raise FeatureExtractionError(f"Expected a 2D feature matrix, got shape {features.shape}")
    if expected_rows is not None and features.shape[0] != expected_rows:
        raise FeatureExtractionError(
            f"Extracted {features.shape[0]} embeddings for {expected_rows} images"
        )
    if features.shape[1] == 0:
        raise FeatureExtractionError("Feature matrix has zero dimensions")

    finite = np.isfinite(features)
    if not finite.all():
        row, column = np.argwhere(~finite)[0]
        raise FeatureExtractionError(
            f"Feature matrix contains {int((~finite).sum())} non-finite values; "
            f"first at row {int(row)}, column {int(column)}"
        )
    if np.allclose(features, 0):
        raise FeatureExtractionError("Feature matrix is all zeros")


def describe_model(name: str = DEFAULT_MODEL_NAME, model: torch.nn.Module | None = None) -> dict:
    """Return identifying information about the checkpoint for reproducibility."""
    description: dict = {
        "name": name,
        "library": "timm",
        "timm_version": timm.__version__,
        "torch_version": torch.__version__,
    }
    if model is not None:
        config = dict(getattr(model, "pretrained_cfg", {}) or {})
        description.update(
            {
                "architecture": type(model).__name__,
                "parameters": int(sum(p.numel() for p in model.parameters())),
                "input_size": list(config.get("input_size", [])),
                "mean": list(config.get("mean", [])),
                "std": list(config.get("std", [])),
                "hf_hub_id": config.get("hf_hub_id"),
            }
        )
    return description
