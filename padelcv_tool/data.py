"""Data utilities for the inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

from .config import BBOX_CSV, CLIP_IMAGE_SIZE, CLIP_NUM_FRAMES


@dataclass(frozen=True)
class ClipSelection:
    video_path: Path
    start_frame: int
    end_frame: int
    bbox_norm: Tuple[float, float, float, float]


def load_class_names(csv_path: Path | None = None) -> List[str]:
    """Derive the stroke label list using the same preprocessing as training."""
    path = csv_path or BBOX_CSV
    if not path.exists():
        return [
            "Backhand",
            "Backhand_Volley",
            "Forehand",
            "Forehand_Volley",
            "Smash",
            "Vibora_Bandeja",
        ]

    df = pd.read_csv(path)
    video_col = df["video"].astype(str)
    strokes = video_col.apply(lambda v: Path(v).parent.name)
    strokes = strokes.replace({"Vibora": "Vibora_Bandeja", "Bandeja": "Vibora_Bandeja"})
    df = df.assign(stroke=strokes)

    class_counts = df["stroke"].value_counts()
    valid = class_counts[class_counts >= 10].index.tolist()
    df = df[df["stroke"].isin(valid)]
    df = df[df["stroke"] != "Lob"]

    return sorted(df["stroke"].unique())


def _clamp_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    x = float(np.clip(x, 0.0, 1.0))
    y = float(np.clip(y, 0.0, 1.0))
    w = float(np.clip(w, 0.01, 1.0))
    h = float(np.clip(h, 0.01, 1.0))
    if x + w > 1.0:
        w = 1.0 - x
    if y + h > 1.0:
        h = 1.0 - y
    return x, y, w, h


def _read_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():  # pragma: no cover - hardware failure
        raise RuntimeError(f"Unable to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Unable to read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def sample_frame_indices(start: int, end: int, target: int) -> np.ndarray:
    total = max(end - start + 1, 1)
    if total >= target:
        return np.linspace(start, end, target, dtype=int)
    indices = list(range(start, end + 1))
    while len(indices) < target:
        indices.append(indices[-1])
    return np.array(indices[:target], dtype=int)


def extract_clip_frames(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    bbox_norm: Tuple[float, float, float, float] | None,
    num_frames: int = CLIP_NUM_FRAMES,
    output_size: int = CLIP_IMAGE_SIZE,
) -> List[np.ndarray]:
    start_frame = max(0, start_frame)
    end_frame = max(start_frame, end_frame)
    bbox = _clamp_bbox(bbox_norm or (0.0, 0.0, 1.0, 1.0))

    indices = sample_frame_indices(start_frame, end_frame, num_frames)
    frames: List[np.ndarray] = []

    # Use a single capture for efficiency when ranges are contiguous
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():  # pragma: no cover
        raise RuntimeError(f"Unable to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x, y, w, h = bbox
    x1 = int(x * width)
    y1 = int(y * height)
    x2 = int((x + w) * width)
    y2 = int((y + h) * height)

    last_read = None
    for idx in indices:
        if last_read is None or idx != last_read + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        last_read = idx
        if not ok:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame
        crop = cv2.resize(crop, (output_size, output_size))
        frames.append(crop)

    cap.release()
    return frames


def frames_to_tensor(frames: Sequence[np.ndarray]) -> torch.Tensor:
    if not frames:
        raise ValueError("No frames provided for tensor conversion")
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2)
    mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
    return (tensor - mean) / std


__all__ = [
    "ClipSelection",
    "extract_clip_frames",
    "frames_to_tensor",
    "load_class_names",
    "sample_frame_indices",
]
