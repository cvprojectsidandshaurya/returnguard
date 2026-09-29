"""Losses for garment embedding training."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def symmetric_infonce(packing_embeddings: torch.Tensor, rider_embeddings: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Cross-domain InfoNCE with packing-to-rider and rider-to-packing terms.

    Rows must be aligned positive pairs and may not repeat a garment in a batch.
    That invariant is enforced by ``HardNegativeBatchSampler``.
    """
    if packing_embeddings.ndim != 2 or rider_embeddings.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if packing_embeddings.shape != rider_embeddings.shape:
        raise ValueError("packing and rider embeddings must have the same shape")
    if packing_embeddings.shape[0] < 2:
        raise ValueError("InfoNCE needs at least two pairs per batch")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    logits = packing_embeddings @ rider_embeddings.T / temperature
    targets = torch.arange(len(logits), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))
