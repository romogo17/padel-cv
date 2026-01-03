"""Pose overlay utilities using RTMO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import cv2
import numpy as np

try:
    from rtmlib import RTMO, draw_skeleton
except Exception:  # pragma: no cover - optional runtime dep
    RTMO = None  # type: ignore
    draw_skeleton = None  # type: ignore


@dataclass
class PoseAnimation:
    frames: List[np.ndarray]
    fps: float


class PoseEstimator:
    def __init__(self, device: str = "cpu", backend: str = "onnxruntime") -> None:
        self.device = device
        self.backend = backend
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None or RTMO is None:
            return
        self._model = RTMO(
            onnx_model="https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip",
            model_input_size=(640, 640),
            backend=self.backend,
            device=self.device,
            score_thr=0.2,
            nms_thr=0.45,
        )

    def render_animation(
        self,
        frames: Sequence[np.ndarray],
        max_frames: int = 30,
        fps: float = 12.0,
    ) -> PoseAnimation | None:
        if RTMO is None or draw_skeleton is None or not frames:
            return None
        self._ensure_model()
        if self._model is None:
            return None

        total = len(frames)
        sample_count = min(max_frames, total)
        indices = np.linspace(0, total - 1, sample_count, dtype=int)
        animated_frames: List[np.ndarray] = []

        for idx in indices:
            frame = frames[idx]
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            keypoints, scores = self._model(bgr)
            keypoints, scores = self._select_primary_actor(keypoints, scores)
            if keypoints.size == 0:
                continue
            h, w, _ = frame.shape
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            canvas[:] = (12, 12, 12)
            skeleton_only = draw_skeleton(canvas, keypoints, scores, kpt_thr=0.3)
            animated_frames.append(skeleton_only)

        if not animated_frames:
            return None

        return PoseAnimation(frames=animated_frames, fps=fps)

    def _select_primary_actor(
        self,
        keypoints: np.ndarray | List[np.ndarray],
        scores: np.ndarray | List[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        keypoints_np = np.asarray(keypoints)
        scores_np = np.asarray(scores)
        if keypoints_np.size == 0 or scores_np.size == 0:
            return np.empty((0, 0, 0)), np.empty((0, 0))
        if keypoints_np.ndim < 3:
            return keypoints_np, scores_np
        persons = keypoints_np.shape[0]
        if persons <= 1:
            return keypoints_np, scores_np
        flat_scores = scores_np.reshape(persons, -1)
        best_idx = int(np.argmax(flat_scores.mean(axis=1)))
        return keypoints_np[best_idx : best_idx + 1], scores_np[best_idx : best_idx + 1]


__all__ = ["PoseEstimator", "PoseAnimation"]
