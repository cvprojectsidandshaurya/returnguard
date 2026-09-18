"""Cross-domain training pairs and leakage-safe hard-negative batches.

The metadata has one row per photo.  A training example deliberately contains
one packing shot and one rider shot of the *same physical garment*.  Batches
contain no repeated garment IDs, so the off-diagonal entries of the InfoNCE
matrix are valid negatives.  When possible, each batch also contains a
different garment with the same design or lookalike group as a hard negative.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import pandas as pd
from torch.utils.data import Dataset, Sampler


def _stable_int(*parts: object) -> int:
    """A process-independent integer for deterministic photo selection."""
    text = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:16], 16)


def _clean_metadata_value(value: object) -> str:
    """Treat pandas' empty-field NaN the same way as an empty CSV field."""
    return "" if pd.isna(value) else str(value).strip()


class CrossDomainPairDataset(Dataset[tuple[Path, Path, str]]):
    """One deterministic packing/rider pair per garment for a given epoch."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        image_root: Path,
        split: str,
        seed: int = 13,
    ) -> None:
        rows = metadata[(metadata["split"] == split) & metadata["shot_type"].isin(["packing", "rider"])]
        self.image_root = image_root
        self.seed = seed
        self.epoch = 0
        self._paths: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self.attributes: dict[str, dict[str, str]] = {}

        for row in rows.itertuples(index=False):
            garment_id = str(row.garment_id)
            self._paths[garment_id][str(row.shot_type)].append(str(row.relative_path))
            self.attributes.setdefault(
                garment_id,
                {
                    "design_id": _clean_metadata_value(row.design_id),
                    "lookalike_group": _clean_metadata_value(row.lookalike_group),
                },
            )

        self.garment_ids = sorted(
            garment_id
            for garment_id, sides in self._paths.items()
            if sides["packing"] and sides["rider"]
        )
        if not self.garment_ids:
            raise ValueError(f"split {split!r} has no garments with both packing and rider photos")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.garment_ids)

    def __getitem__(self, index: int) -> tuple[Path, Path, str]:
        garment_id = self.garment_ids[index]
        sides = self._paths[garment_id]
        packing = sides["packing"][_stable_int(self.seed, self.epoch, garment_id, "packing") % len(sides["packing"])]
        rider = sides["rider"][_stable_int(self.seed, self.epoch, garment_id, "rider") % len(sides["rider"])]
        return self.image_root / packing, self.image_root / rider, garment_id


class HardNegativeBatchSampler(Sampler[list[int]]):
    """Build unique-garment batches, prioritising design and lookalike negatives."""

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
        for index, garment_id in enumerate(self.dataset.garment_ids):
            attrs = self.dataset.attributes[garment_id]
            if attrs["design_id"]:
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
    required = {"garment_id", "design_id", "lookalike_group", "shot_type", "relative_path", "split"}
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"metadata is missing required columns: {missing}")
    if df.empty:
        raise ValueError("metadata.csv is empty, capture garments before training")
    return df
