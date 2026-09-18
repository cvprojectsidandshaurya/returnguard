"""Trainable DINOv2 encoder for cross-domain garment retrieval."""

from __future__ import annotations

import torch
from torch import nn

from .embed import BACKBONES


class DinoMetricModel(nn.Module):
    """DINOv2 plus a small projection head, returning L2-normalised embeddings."""

    def __init__(self, backbone_name: str = "dinov2_base", projection_dim: int = 512, train_last_blocks: int = 2) -> None:
        super().__init__()
        if backbone_name not in BACKBONES or BACKBONES[backbone_name]["kind"] != "hf":
            raise ValueError("fine tuning currently supports dinov2_base or dinov2_large")
        if train_last_blocks < 0:
            raise ValueError("train_last_blocks must be zero or positive")

        from transformers import AutoImageProcessor, AutoModel

        spec = BACKBONES[backbone_name]
        self.backbone_name = backbone_name
        self.processor = AutoImageProcessor.from_pretrained(spec["id"])
        self.backbone = AutoModel.from_pretrained(spec["id"])
        hidden_size = self.backbone.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, projection_dim),
        )
        self._freeze_to_last_blocks(train_last_blocks)

    def _freeze_to_last_blocks(self, train_last_blocks: int) -> None:
        """Keep the projection head trainable and optionally tune DINO's final blocks."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if train_last_blocks == 0:
            return
        layers = list(getattr(getattr(self.backbone, "encoder", None), "layer", []))
        if not layers:
            raise RuntimeError("could not find transformer blocks on the selected DINO backbone")
        for layer in layers[-train_last_blocks:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        # The final layer norm adapts the representation scale before projection.
        for name, parameter in self.backbone.named_parameters():
            if "layernorm" in name.lower():
                parameter.requires_grad = True

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.backbone(pixel_values=pixel_values)
        cls = output.last_hidden_state[:, 0]
        return nn.functional.normalize(self.projector(cls), dim=-1)

    def trainable_parameter_groups(self, head_lr: float, backbone_lr_scale: float = 0.1) -> list[dict]:
        head = [p for p in self.projector.parameters() if p.requires_grad]
        backbone = [p for p in self.backbone.parameters() if p.requires_grad]
        groups = [{"params": head, "lr": head_lr}]
        if backbone:
            groups.append({"params": backbone, "lr": head_lr * backbone_lr_scale})
        return groups
