"""Frozen DINOv2 feature extractor with explicit spatial pooling.

The backbone is the official DINOv2 checkpoint distributed on the Hugging Face
Hub and loaded through :mod:`transformers`, which is what the RoVF reference
implementation does (``AutoModel.from_pretrained("facebook/dinov2-small")``).
It is frozen: no parameter is ever trained here.

Token layout
------------
DINOv2 returns ``last_hidden_state`` of shape ``(batch, 1 + patches, hidden)``.
Index 0 is the CLS token and the remaining 256 positions are the 16x16 patch
grid for a 224x224 input with patch size 14. Three spatial reductions are
supported and must not be conflated:

``patches``
    Mean of the patch tokens only (``[:, 1:]``). This is the "patch-token
    spatial mean pooling" of the task description.
``all``
    Mean of every token including CLS. This is what the RoVF
    ``forward_strat="average"`` does (``torch.mean(stacked, dim=-2)`` over the
    full sequence), even though the paper prose says "patch embeddings".
``cls``
    The CLS token only, which is what RoVF uses for its fine-tuned
    ``imagedino_*`` configs (``forward_strat: cls``) and for the frame-based
    DINOv2 baselines.

Keeping all three lets one extraction pass answer both "what does the correct
DINOv2 pooling give" and "what did the paper actually run", without a second
forward pass over the images.

Preprocessing
-------------
``imagenet``
    DINOv2's own recipe: resize to 224x224, scale to [0, 1], normalize with the
    ImageNet mean and standard deviation that the checkpoint was trained with.
``rovf``
    What the reference evaluation loop does: resize to 224x224, divide by 255,
    and *no* normalization (``AnimalClipDataset.load_clip`` in the RoVF
    repository). It is kept because it is required to compare against the
    published number, and it is recorded in the run metadata.

PolarBearVidID frames are 256x128. Both recipes resize directly to a square,
which is a horizontal squeeze; the official RoVF conversion does the same with
``cv2.resize(..., (224, 224))``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from transformers import AutoModel

from wildlife_tools.data import WildlifeDataset
from wildlife_tools.features import DeepFeatures

DEFAULT_MODEL_NAME = "facebook/dinov2-small"
DEFAULT_IMAGE_SIZE = 224
SPATIAL_REDUCTIONS = ("patches", "all", "cls")
DEFAULT_REDUCTION = "patches"
PREPROCESSING_MODES = ("imagenet", "rovf")
DEFAULT_PREPROCESSING = "imagenet"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FeatureExtractionError(RuntimeError):
    """Raised when extracted features are structurally invalid."""


def _import_transformers_version() -> str:
    import transformers

    return transformers.__version__


def load_model(name: str = DEFAULT_MODEL_NAME, device: str = "cpu") -> torch.nn.Module:
    """Load a frozen, evaluation-mode DINOv2 backbone."""
    model = AutoModel.from_pretrained(name)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device)


def build_transform(
    image_size: int = DEFAULT_IMAGE_SIZE,
    preprocessing: str = DEFAULT_PREPROCESSING,
) -> T.Compose:
    """Build the per-model preprocessing pipeline."""
    if preprocessing not in PREPROCESSING_MODES:
        raise FeatureExtractionError(
            f"Unknown preprocessing {preprocessing!r}; expected one of {PREPROCESSING_MODES}"
        )
    if image_size <= 0:
        raise FeatureExtractionError(f"Invalid image size {image_size}")

    if preprocessing == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:  # "rovf": the reference loader only divides by 255.
        mean, std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)

    return T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def spatial_pool(tokens: torch.Tensor, reduction: str = DEFAULT_REDUCTION) -> torch.Tensor:
    """Reduce ``(batch, tokens, hidden)`` to ``(batch, hidden)``.

    ``patches`` excludes the CLS token at position 0; ``all`` includes it;
    ``cls`` keeps only it. These are different features and the distinction is
    the whole point of this function.
    """
    if tokens.ndim != 3:
        raise FeatureExtractionError(
            f"Expected (batch, tokens, hidden) tokens, got shape {tuple(tokens.shape)}"
        )
    if tokens.shape[1] < 2:
        raise FeatureExtractionError("Token sequence is too short to separate CLS from patches")

    if reduction == "patches":
        return tokens[:, 1:, :].mean(dim=1)
    if reduction == "all":
        return tokens.mean(dim=1)
    if reduction == "cls":
        return tokens[:, 0, :]
    raise FeatureExtractionError(
        f"Unknown spatial reduction {reduction!r}; expected one of {SPATIAL_REDUCTIONS}"
    )


class SpatialPooler(torch.nn.Module):
    """Wrap a DINOv2 backbone and reduce every frame's tokens.

    The forward pass returns the concatenation of the requested reductions, so
    a single image pass yields several embedding variants. The extraction code
    reshapes the result back to ``(rows, reductions, hidden)``.
    """

    def __init__(self, backbone: torch.nn.Module, reductions: tuple[str, ...]) -> None:
        super().__init__()
        unknown = [reduction for reduction in reductions if reduction not in SPATIAL_REDUCTIONS]
        if unknown:
            raise FeatureExtractionError(
                f"Unknown spatial reductions {unknown}; expected one of {SPATIAL_REDUCTIONS}"
            )
        if not reductions:
            raise FeatureExtractionError("At least one spatial reduction is required")
        self.backbone = backbone
        self.reductions = tuple(reductions)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise FeatureExtractionError(
                f"Expected (batch, 3, height, width) images, got {tuple(pixel_values.shape)}"
            )
        tokens = self.backbone(pixel_values=pixel_values).last_hidden_state
        return torch.cat([spatial_pool(tokens, reduction) for reduction in self.reductions], dim=1)


def extract_frame_features(
    model: torch.nn.Module,
    transform: T.Compose,
    frames: pd.DataFrame,
    reductions: tuple[str, ...] = (DEFAULT_REDUCTION,),
    batch_size: int = 32,
    device: str = "cpu",
    num_workers: int = 0,
) -> np.ndarray:
    """Embed every frame of ``frames`` once, for every requested reduction.

    Returns:
        ``(len(frames), len(reductions), hidden)`` float32 array, row-aligned
        with ``frames``.
    """
    if frames.empty:
        raise FeatureExtractionError("No frames to embed")

    hidden = int(getattr(model.config, "hidden_size", 0))
    if hidden <= 0:
        raise FeatureExtractionError("Backbone does not expose config.hidden_size")

    dataset = WildlifeDataset(
        frames.reset_index(drop=True),
        root=None,
        transform=transform,
        img_load="full",
        load_label=True,
    )
    extractor = DeepFeatures(
        model=SpatialPooler(model, tuple(reductions)),
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    raw = np.asarray(extractor(dataset), dtype="float32")
    features = raw.reshape(len(frames), len(reductions), hidden)
    validate_frame_features(
        features,
        expected_rows=len(frames),
        expected_dim=hidden,
        expected_reductions=len(reductions),
    )
    return features


def validate_frame_features(
    features: np.ndarray,
    expected_rows: int | None = None,
    expected_dim: int | None = None,
    expected_reductions: int | None = None,
) -> None:
    """Reject non-finite, all-zero or wrongly shaped frame embeddings."""
    features = np.asarray(features)
    if features.ndim != 3:
        raise FeatureExtractionError(
            f"Expected (frames, reductions, dim), got shape {features.shape}"
        )
    if expected_rows is not None and features.shape[0] != expected_rows:
        raise FeatureExtractionError(
            f"Extracted {features.shape[0]} frames for {expected_rows} inputs"
        )
    if expected_reductions is not None and features.shape[1] != expected_reductions:
        raise FeatureExtractionError(
            f"Extracted {features.shape[1]} reductions for {expected_reductions} requested"
        )
    if expected_dim is not None and features.shape[2] != expected_dim:
        raise FeatureExtractionError(
            f"Embedding dimension {features.shape[2]} does not match {expected_dim}"
        )
    if features.shape[2] == 0:
        raise FeatureExtractionError("Embedding has zero dimensions")

    finite = np.isfinite(features)
    if not finite.all():
        row, reduction, column = np.argwhere(~finite)[0]
        raise FeatureExtractionError(
            f"Frame features contain {int((~finite).sum())} non-finite values; "
            f"first at frame {int(row)}, reduction {int(reduction)}, dim {int(column)}"
        )
    if np.allclose(features, 0):
        raise FeatureExtractionError("Frame features are all zeros")


def l2_normalize(features: np.ndarray) -> np.ndarray:
    """L2-normalize the last axis, rejecting zero-norm embeddings."""
    features = np.asarray(features, dtype="float32")
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise FeatureExtractionError("Cannot normalize zero-norm embeddings")
    return (features / norms).astype("float32")


def checkpoint_revision(
    name: str = DEFAULT_MODEL_NAME, config: object | None = None
) -> str | None:
    """Return the exact Hub revision of a downloaded checkpoint.

    ``transformers`` 5.x no longer populates ``config._commit_hash``, so the
    revision is read from the local Hub cache (``refs/main``). Both are offline
    lookups; a locally supplied path simply has no Hub revision.
    """
    commit = getattr(config, "_commit_hash", None) if config is not None else None
    if commit:
        return str(commit)
    if not name or Path(name).expanduser().exists():
        return None
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        folder = Path(HF_HUB_CACHE) / f"models--{str(name).replace('/', '--')}"
        reference = folder / "refs" / "main"
        if reference.is_file():
            return reference.read_text(encoding="utf-8").strip() or None
    except (ImportError, OSError):
        return None
    return None


def describe_model(name: str = DEFAULT_MODEL_NAME, model: torch.nn.Module | None = None) -> dict:
    """Return identifying information about the checkpoint for reproducibility."""
    description: dict = {
        "name": name,
        "library": "transformers",
        "transformers_version": _import_transformers_version(),
        "torch_version": torch.__version__,
        "checkpoint": name,
        "checkpoint_revision": checkpoint_revision(name),
    }
    config = getattr(model, "config", None)
    if config is not None:
        description.update(
            {
                "checkpoint": str(getattr(config, "_name_or_path", name) or name),
                "checkpoint_revision": checkpoint_revision(name, config),
                "config_version": getattr(config, "transformers_version", None),
                "architecture": type(model).__name__,
                "parameters": int(sum(p.numel() for p in model.parameters())),
                "hidden_size": int(getattr(config, "hidden_size", 0)),
                "patch_size": int(getattr(config, "patch_size", 0)),
                "image_size": int(getattr(config, "image_size", 0)),
                "num_hidden_layers": int(getattr(config, "num_hidden_layers", 0)),
                "frozen": all(not p.requires_grad for p in model.parameters()),
            }
        )
    return description
