"""Inference wrapper for a checkpoint trained by ``scripts/train_metric.py``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .embed import pick_device
from .metric_model import DinoMetricModel


class FineTunedEmbedder:
    """Load a saved retrieval checkpoint and encode paths into unit vectors."""

    def __init__(self, checkpoint: Path | str, device: str = "auto") -> None:
        checkpoint_path = Path(checkpoint)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "model" not in payload or "run" not in payload:
            raise ValueError(f"{checkpoint_path} is not a ReturnGuard metric-learning checkpoint")
        run = payload["run"]
        required = ("backbone", "projection_dim", "train_last_blocks")
        missing = [field for field in required if field not in run]
        if missing:
            raise ValueError(f"checkpoint run metadata is missing: {missing}")

        self.device = pick_device(device)
        self.model = DinoMetricModel(
            backbone_name=run["backbone"],
            projection_dim=int(run["projection_dim"]),
            train_last_blocks=int(run["train_last_blocks"]),
        )
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(self.device).eval()
        self.checkpoint = checkpoint_path

    @torch.no_grad()
    def encode(self, paths: list[Path], batch_size: int = 16) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        vectors: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            images: list[Image.Image] = []
            for path in chunk:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            pixels = self.model.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
            vectors.append(self.model(pixels).cpu().numpy())
        if not vectors:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(vectors, axis=0).astype(np.float32)
