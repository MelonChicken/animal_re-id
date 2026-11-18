"""Load and validate an extracted PolarBearVidID dataset."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from PIL import Image
from wildlife_datasets.datasets import PolarBearVidID


def load_polar_bear_vid_id(root: str | Path) -> pd.DataFrame:
    """Return the WildlifeDatasets catalogue with absolute image paths.

    Point root at the directory containing animal_db.csv and the numbered
    identity folders. The catalogue is built without decoding images.
    WildlifeDatasets discovers frames on disk and supplies the video IDs.
    Use the identity and video columns together to identify a sequence.
    """
    root = Path(root).expanduser().resolve()
    metadata_path = root / "animal_db.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing {metadata_path}. Use the extracted PolarBearVidID directory "
            "that directly contains animal_db.csv and the numbered identity folders."
        )

    try:
        catalogue = PolarBearVidID(
            str(root),
            check_files=False,
            update_wrong_labels=False,
        ).df
    except Exception as error:
        # WildlifeDatasets also raises plain Exception for catalogue errors.
        raise ValueError(
            f"Cannot build the PolarBearVidID catalogue at {root}. "
            f"Check animal_db.csv and the numbered image folders: {error}"
        ) from error

    required_columns = {"image_id", "path", "identity"}
    missing_columns = required_columns.difference(catalogue.columns)
    if missing_columns or catalogue.empty:
        raise ValueError(
            f"Invalid PolarBearVidID catalogue at {root}: {len(catalogue)} frames "
            f"and missing required columns {sorted(missing_columns)}."
        )

    paths = catalogue["path"]
    if paths.isna().any() or paths.str.strip().eq("").any():
        raise ValueError(f"PolarBearVidID catalogue at {root} contains empty image paths.")
    catalogue["path"] = paths.map(lambda path: os.path.abspath(root / path))

    missing = catalogue.loc[~catalogue["path"].map(os.path.isfile), "path"]
    if not missing.empty:
        raise FileNotFoundError(
            f"{len(missing)} referenced PolarBearVidID image files are missing. "
            f"First missing file: {missing.iloc[0]}"
        )
    return catalogue


def _validate_catalogue(catalogue: pd.DataFrame, decode_samples: int) -> dict:
    identity = catalogue["identity"].astype("string").str.strip()
    invalid_identity = (
        identity.isna()
        | identity.eq("")
        | identity.str.casefold().eq(PolarBearVidID.unknown_name)
    )
    valid = catalogue.loc[~invalid_identity]
    normalized_paths = catalogue["path"].map(os.path.normcase)
    stats = {
        "frames": len(catalogue),
        "individuals": valid["identity"].nunique(),
        "images_per_identity": valid["identity"].value_counts().sort_index().to_dict(),
        "missing_referenced_images": 0,  # Checked by the loader.
        "duplicate_path_rows": int(normalized_paths.duplicated(keep=False).sum()),
        "invalid_identity_rows": int(invalid_identity.sum()),
    }
    if "video" in catalogue:
        sequences = valid[["identity", "video"]].dropna().drop_duplicates()
        bare_video_ids = sequences["video"].nunique()
        reused_video_ids = int(sequences["video"].value_counts().gt(1).sum())
        stats.update(
            sequences=len(sequences),
            sequences_per_identity=sequences["identity"].value_counts().sort_index().to_dict(),
            distinct_video_ids=bare_video_ids,
            video_ids_globally_unique=reused_video_ids == 0,
            video_ids_reused_across_identities=reused_video_ids,
            missing_video_rows=int(catalogue["video"].isna().sum()),
        )

    paths = catalogue["path"].sort_values().drop_duplicates().reset_index(drop=True)
    sample_count = min(decode_samples, len(paths))
    decoded = []
    decode_failures = []
    if sample_count:
        indices = [i * len(paths) // sample_count for i in range(sample_count)]
        for path in paths.iloc[indices]:
            try:
                with Image.open(path) as image:
                    image.load()
                    decoded.append({"path": path, "size": list(image.size)})
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                decode_failures.append({"path": path, "error": str(error)})
    stats["decoded_samples"] = decoded
    stats["decode_failures"] = decode_failures
    return stats


def _compare_track_manifest(root: Path, catalogue: pd.DataFrame) -> dict:
    """Compare observed frame counts with the release's inclusive frame ranges."""
    manifest = root / "track_info.csv"
    if not manifest.is_file() or "video" not in catalogue:
        return {"track_manifest_checked": False}

    try:
        track_columns = ["id", "tracklet", "start", "end"]
        tracks = pd.read_csv(manifest, usecols=track_columns, dtype="Int64")
        animals = pd.read_csv(root / "animal_db.csv", usecols=["id", "name"])
        tracks = tracks.merge(animals, on="id", how="left", validate="many_to_one")
        tracks = tracks.rename(columns={"name": "identity", "tracklet": "video"})
        if (
            tracks.empty
            or tracks.isna().any().any()
            or tracks.duplicated(["identity", "video"]).any()
            or tracks["start"].lt(0).any()
            or tracks["end"].lt(tracks["start"]).any()
        ):
            raise ValueError("Expected unique tracks with known identities and valid frame ranges.")
    except ValueError as error:
        raise ValueError(
            f"Cannot validate PolarBearVidID track counts from {manifest}: {error}"
        ) from error

    tracks = tracks.set_index(["identity", "video"])
    expected = tracks["end"] - tracks["start"] + 1
    observed = catalogue.value_counts(["identity", "video"])
    difference = expected.subtract(observed, fill_value=0)
    return {
        "track_manifest_checked": True,
        "frames_missing_from_declared_tracks": int(difference.clip(lower=0).sum()),
        "frames_beyond_declared_tracks": int(-difference.clip(upper=0).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an extracted PolarBearVidID dataset.")
    parser.add_argument("root", type=Path, help="Directory containing animal_db.csv")
    parser.add_argument("--decode-samples", type=int, default=5, help="Number of frames to decode")
    args = parser.parse_args()
    if args.decode_samples < 0:
        parser.error("--decode-samples must be nonnegative")
    try:
        catalogue = load_polar_bear_vid_id(args.root)
        manifest_stats = _compare_track_manifest(args.root.expanduser().resolve(), catalogue)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    stats = _validate_catalogue(catalogue, args.decode_samples)
    stats.update(manifest_stats)
    print(json.dumps(stats, indent=2))
    return int(
        bool(
            stats["duplicate_path_rows"]
            or stats["invalid_identity_rows"]
            or stats.get("missing_video_rows")
            or stats.get("frames_missing_from_declared_tracks")
            or stats.get("frames_beyond_declared_tracks")
            or stats["decode_failures"]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
