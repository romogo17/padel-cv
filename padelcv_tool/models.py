"""Model wrappers and inference orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchvideo.models.hub import i3d_r50
from transformers import TimesformerForVideoClassification

from .config import CLIP_NUM_FRAMES, GCS_WEIGHTS, WEIGHTS_DIR
from .gcs_utils import GCSError, download_from_gcs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class PredictionResult:
    model_name: str
    probabilities: List[Tuple[str, float]]
    inference_ms: float


class BaseStrokeModel:
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.model: nn.Module | None = None

    def predict(self, clip_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class I3DStrokeModel(BaseStrokeModel):
    def __init__(self, class_names: List[str], weights_path: Path):
        super().__init__(class_names)
        net = i3d_r50(pretrained=False)
        in_features = net.blocks[-1].proj.in_features
        net.blocks[-1].proj = nn.Sequential(
            nn.Dropout(0.6), nn.Linear(in_features, len(class_names))
        )
        state = torch.load(weights_path, map_location="cpu")
        net.load_state_dict(state)
        self.model = net.to(DEVICE).eval()

    def predict(self, clip_tensor: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("I3D model is not initialized")
        with torch.no_grad():
            logits = self.model(clip_tensor.to(DEVICE))
            return F.softmax(logits, dim=1)


class TimeSformerStrokeModel(BaseStrokeModel):
    def __init__(self, class_names: List[str], weights_path: Path):
        super().__init__(class_names)
        net = TimesformerForVideoClassification.from_pretrained(
            "facebook/timesformer-base-finetuned-k400"
        )
        net.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(net.config.hidden_size, len(class_names))
        )
        net.config.num_frames = CLIP_NUM_FRAMES
        state = torch.load(weights_path, map_location="cpu")
        net.load_state_dict(state)
        self.model = net.to(DEVICE).eval()

    def predict(self, clip_tensor: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("TimeSformer model is not initialized")
        with torch.no_grad():
            clip = clip_tensor.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
            outputs = self.model(pixel_values=clip.to(DEVICE))
            return F.softmax(outputs.logits, dim=1)


class StrokeInferenceEngine:
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self._models: Dict[str, BaseStrokeModel] = {}
        self._weights = {
            "i3d": WEIGHTS_DIR / "i3d_final.pth",
            "timesformer": WEIGHTS_DIR / "timesformer_final.pth",
        }

    def _ensure_weights(self, key: str) -> Path:
        local_path = self._weights[key]
        if local_path.exists():
            return local_path
        uri = GCS_WEIGHTS.get(key)
        if not uri:
            raise FileNotFoundError(f"No weight URI registered for {key}")
        try:
            return download_from_gcs(uri, local_path)
        except GCSError as exc:
            raise FileNotFoundError(
                f"Weights for {key} are missing at {local_path} and could not be downloaded."
            ) from exc

    def _load_model(self, key: str) -> BaseStrokeModel:
        if key in self._models:
            return self._models[key]
        weights = self._ensure_weights(key)
        if key == "i3d":
            model = I3DStrokeModel(self.class_names, weights)
        elif key == "timesformer":
            model = TimeSformerStrokeModel(self.class_names, weights)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported model key: {key}")
        self._models[key] = model
        return model

    def predict(self, key: str, clip_tensor: torch.Tensor) -> PredictionResult:
        if clip_tensor.dim() != 4:
            raise ValueError("clip_tensor must have shape (C, T, H, W)")
        batch_tensor = clip_tensor.unsqueeze(0)
        model = self._load_model(key)
        start = perf_counter()
        probs = model.predict(batch_tensor)
        elapsed = (perf_counter() - start) * 1000.0
        probs_cpu = probs.squeeze(0).cpu().tolist()
        pairs = sorted(zip(self.class_names, probs_cpu), key=lambda item: item[1], reverse=True)
        return PredictionResult(model_name=key, probabilities=pairs, inference_ms=elapsed)


__all__ = ["StrokeInferenceEngine", "PredictionResult", "DEVICE"]
