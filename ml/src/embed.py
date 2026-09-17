"""Zero shot image embedding backbones.

Every backbone returns L2 normalised float32 vectors, so cosine similarity is a
plain dot product. Add a backbone here, never inline in an eval script, so the
same weights and preprocessing are used everywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

BACKBONES = {
    "dinov2_base": {"kind": "hf", "id": "facebook/dinov2-base"},
    "dinov2_large": {"kind": "hf", "id": "facebook/dinov2-large"},
    "clip_vit_b32": {"kind": "open_clip", "id": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"},
    "clip_vit_l14": {"kind": "open_clip", "id": "ViT-L-14", "pretrained": "laion2b_s32b_b82k"},
}


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    """Wraps one backbone. Call `encode` with a list of image paths."""

    def __init__(self, name: str, device: str = "auto") -> None:
        if name not in BACKBONES:
            raise ValueError(f"unknown backbone {name!r}, options: {sorted(BACKBONES)}")
        self.name = name
        self.spec = BACKBONES[name]
        self.device = pick_device(device)
        self._load()

    def _load(self) -> None:
        kind = self.spec["kind"]
        if kind == "hf":
            from transformers import AutoImageProcessor, AutoModel

            self.processor = AutoImageProcessor.from_pretrained(self.spec["id"])
            self.model = AutoModel.from_pretrained(self.spec["id"]).to(self.device).eval()
        elif kind == "open_clip":
            import open_clip

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.spec["id"], pretrained=self.spec["pretrained"]
            )
            self.model = self.model.to(self.device).eval()
        else:
            raise ValueError(f"unknown backbone kind {kind!r}")

    def _forward(self, images: list[Image.Image]) -> torch.Tensor:
        if self.spec["kind"] == "hf":
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            out = self.model(**inputs)
            # CLS token. Mean pooling of patch tokens is a reasonable alternative,
            # but pick one and keep it fixed or the numbers stop being comparable.
            return out.last_hidden_state[:, 0]
        batch = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        return self.model.encode_image(batch)

    @torch.no_grad()
    def encode(self, paths: list[Path], batch_size: int = 16) -> np.ndarray:
        vectors = []
        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            images = [Image.open(p).convert("RGB") for p in chunk]
            feats = self._forward(images).float()
            feats = torch.nn.functional.normalize(feats, dim=-1)
            vectors.append(feats.cpu().numpy())
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(vectors, axis=0).astype(np.float32)
