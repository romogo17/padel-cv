"""Global configuration helpers for the inference GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATASET_ROOT: Final[Path] = PROJECT_ROOT / "dataset"
ARTIFACTS_ROOT: Final[Path] = PROJECT_ROOT / "artifacts"
PREPROCESSED_CLIPS: Final[Path] = ARTIFACTS_ROOT / "preprocessed_video_clips"
POSE_CACHE_DIR: Final[Path] = ARTIFACTS_ROOT / "poses_bbox"
WEIGHTS_DIR: Final[Path] = ARTIFACTS_ROOT / "pretrained"
DEFAULT_MAIN_VIDEO_DIR: Final[Path] = PROJECT_ROOT / "dataset"

BBOX_CSV: Final[Path] = DATASET_ROOT / "bboxes.csv"
CLIP_NUM_FRAMES: Final[int] = 20
CLIP_IMAGE_SIZE: Final[int] = 224

GCS_WEIGHTS: Final[dict[str, str]] = {
    "i3d": "gs://padel-cv-dataset/weights/i3d_final.pth",
    "timesformer": "gs://padel-cv-dataset/weights/timesformer_final.pth",
}


def ensure_artifact_dirs() -> None:
    """Make sure artifact directories exist before writing files."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_CLIPS.mkdir(parents=True, exist_ok=True)


__all__ = [
    "PROJECT_ROOT",
    "DATASET_ROOT",
    "ARTIFACTS_ROOT",
    "PREPROCESSED_CLIPS",
    "POSE_CACHE_DIR",
    "WEIGHTS_DIR",
    "DEFAULT_MAIN_VIDEO_DIR",
    "BBOX_CSV",
    "CLIP_NUM_FRAMES",
    "CLIP_IMAGE_SIZE",
    "GCS_WEIGHTS",
    "ensure_artifact_dirs",
]
