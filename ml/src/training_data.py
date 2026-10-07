"""Cross-domain product/unit identity pairs and leakage-safe hard negatives.

The metadata's ``garment_id`` is globally unique per physical item; ``unit_id``
is only a number *within a design* and must never be used as a global key.
Product matching groups by ``design_id``. Unit matching groups by
``garment_id`` and intentionally treats identical designs as hard negatives.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import pandas as pd
from torch.utils.data import Dataset, Sampler


IDENTITY_COLUMNS = {"product": "design_id", "unit": "garment_id"}


def identity_column(identity_policy: str) -> str:
    try:
        return IDENTITY_COLUMNS[identity_policy]
    except KeyError as exc:
        raise ValueError(f"identity_policy must be one of {sorted(IDENTITY_COLUMNS)}") from exc


def _stable_int(*parts: object) -> int:
    """A process-independent integer for deterministic photo selection."""
    text = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:16], 16)


def _clean_metadata_value(value: object) -> str:
    """Treat pandas' empty-field NaN the same way as an empty CSV field."""
    return "" if pd.isna(value) else str(value).strip()


class CrossDomainPairDataset(Dataset[tuple[Path, Path, str]]):
    """One deterministic packing/rider pair per configured identity per epoch."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        image_root: Path,
        split: str,
        seed: int = 13,
        identity_policy: str = "product",
    ) -> None:
        rows = metadata[(metadata["split"] == split) & metadata["shot_type"].isin(["packing", "rider"])]
        self.image_root = image_root
        self.seed = seed
        self.epoch = 0
        self.identity_policy = identity_policy
        self.identity_column = identity_column(identity_policy)
        self._paths: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self.attributes: dict[str, dict[str, str]] = {}

        for row in rows.itertuples(index=False):
            identity_id = _clean_metadata_value(getattr(row, self.identity_column))
            if not identity_id:
                raise ValueError(f"{self.identity_column} must be populated for {identity_policy}-identity training")
            self._paths[identity_id][str(row.shot_type)].append(str(row.relative_path))
            self.attributes.setdefault(
                identity_id,
                {
                    "design_id": _clean_metadata_value(row.design_id),
                    "lookalike_group": _clean_metadata_value(row.lookalike_group),
                },
            )

        self.identity_ids = sorted(
            identity_id
            for identity_id, sides in self._paths.items()
            if sides["packing"] and sides["rider"]
        )
        if not self.identity_ids:
            raise ValueError(f"split {split!r} has no {identity_policy} identities with both shot types")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.identity_ids)

    def __getitem__(self, index: int) -> tuple[Path, Path, str]:
        identity_id = self.identity_ids[index]
        sides = self._paths[identity_id]
        packing = sides["packing"][_stable_int(self.seed, self.epoch, identity_id, "packing") % len(sides["packing"])]
        rider = sides["rider"][_stable_int(self.seed, self.epoch, identity_id, "rider") % len(sides["rider"])]
        return self.image_root / packing, self.image_root / rider, identity_id


class HardNegativeBatchSampler(Sampler[list[int]]):
    """Build unique-identity batches, prioritising valid hard negatives."""

    def __init__(
        self,
        dataset: CrossDomainPairDataset,
        batch_size: int,
        seed: int = 13,
        drop_last: bool = True,
    ) -> None:
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2 for contrastive learning")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self._hard_neighbors = self._build_hard_neighbors()

    def _build_hard_neighbors(self) -> dict[int, set[int]]:
        by_design: dict[str, set[int]] = defaultdict(set)
        by_lookalike: dict[str, set[int]] = defaultdict(set)
        for index, identity_id in enumerate(self.dataset.identity_ids):
            attrs = self.dataset.attributes[identity_id]
            if self.dataset.identity_policy == "unit" and attrs["design_id"]:
                by_design[attrs["design_id"]].add(index)
            if attrs["lookalike_group"]:
                by_lookalike[attrs["lookalike_group"]].add(index)

        neighbors: dict[int, set[int]] = defaultdict(set)
        for members in [*by_design.values(), *by_lookalike.values()]:
            for index in members:
                neighbors[index].update(members - {index})
        return neighbors

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        remaining = set(range(len(self.dataset)))
        while remaining:
            anchor = rng.choice(sorted(remaining))
            batch = [anchor]
            remaining.remove(anchor)

            hard = sorted(self._hard_neighbors.get(anchor, set()) & remaining)
            if hard:
                choice = rng.choice(hard)
                batch.append(choice)
                remaining.remove(choice)

            fill = min(self.batch_size - len(batch), len(remaining))
            if fill:
                extras = rng.sample(sorted(remaining), fill)
                batch.extend(extras)
                remaining.difference_update(extras)

            if len(batch) == self.batch_size or not self.drop_last:
                if len(batch) >= 2:
                    yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def assert_image_paths_exist(dataset: CrossDomainPairDataset) -> None:
    """Fail before the first model download if metadata and image storage disagree."""
    missing: list[Path] = []
    for index in range(len(dataset)):
        packing, rider, _ = dataset[index]
        missing.extend(path for path in (packing, rider) if not path.is_file())
        if len(missing) >= 10:
            break
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(f"image paths in metadata are missing under image_root: {preview}")


def load_metadata(path: Path) -> pd.DataFrame:
    required = {"garment_id", "unit_id", "design_id", "lookalike_group", "shot_type", "relative_path", "split"}
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"metadata is missing required columns: {missing}")
    if df.empty:
        raise ValueError("metadata.csv is empty, capture garments before training")
    return df
