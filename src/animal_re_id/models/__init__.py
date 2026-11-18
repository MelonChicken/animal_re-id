"""Frozen animal re-identification baselines."""

from animal_re_id.models.megadescriptor import (
    DEFAULT_MODEL_NAME,
    build_transform,
    describe_model,
    extract_features,
    load_model,
)

__all__ = [
    "DEFAULT_MODEL_NAME",
    "build_transform",
    "describe_model",
    "extract_features",
    "load_model",
]
