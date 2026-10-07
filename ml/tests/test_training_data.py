"""Tests for unit-identity pair construction and hard-negative sampling."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training_data import CrossDomainPairDataset, HardNegativeBatchSampler  # noqa: E402


def _rows(unit_id: str, design_id: str) -> list[dict]:
    return [
        {"garment_id": f"g-{unit_id}", "unit_id": unit_id, "design_id": design_id, "lookalike_group": "", "shot_type": "packing", "relative_path": f"{unit_id}/p.jpg", "split": "train"},
        {"garment_id": f"g-{unit_id}", "unit_id": unit_id, "design_id": design_id, "lookalike_group": "", "shot_type": "rider", "relative_path": f"{unit_id}/r.jpg", "split": "train"},
    ]


def test_same_design_physical_garments_are_unit_hard_negatives():
    data = pd.DataFrame(_rows("u-1", "d-1") + _rows("u-2", "d-1") + _rows("u-3", "d-2"))
    dataset = CrossDomainPairDataset(data, Path("/images"), split="train", identity_policy="unit")
    sampler = HardNegativeBatchSampler(dataset, batch_size=2, seed=13, drop_last=True)
    first, second = dataset.identity_ids.index("g-u-1"), dataset.identity_ids.index("g-u-2")
    assert second in sampler._hard_neighbors[first]
    assert first in sampler._hard_neighbors[second]


def test_product_identity_groups_same_design_without_using_local_unit_number():
    data = pd.DataFrame(_rows("1", "d-1") + _rows("1", "d-2"))
    dataset = CrossDomainPairDataset(data, Path("/images"), split="train", identity_policy="product")
    assert dataset.identity_ids == ["d-1", "d-2"]
