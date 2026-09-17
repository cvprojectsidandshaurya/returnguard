#!/usr/bin/env python3
"""Assign train/val/test in data/metadata.csv by split group, with a fixed seed.

Never assign splits by hand and never split at the image level. The unit here
is the connected component over shared design_id and shared lookalike_group.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.splits import split_groups  # noqa: E402


def assign(df: pd.DataFrame, seed: int, ratios: tuple[float, float, float]) -> pd.DataFrame:
    groups = split_groups(df)
    roots = sorted(set(groups.values))

    rng = random.Random(seed)
    rng.shuffle(roots)

    n = len(roots)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    assignment = {}
    for i, root in enumerate(roots):
        if i < n_train:
            assignment[root] = "train"
        elif i < n_train + n_val:
            assignment[root] = "val"
        else:
            assignment[root] = "test"

    out = df.copy()
    out["split"] = out["garment_id"].map(groups).map(assignment)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train", type=float, default=0.6)
    parser.add_argument("--val", type=float, default=0.2)
    args = parser.parse_args()

    test = round(1.0 - args.train - args.val, 6)
    if test <= 0:
        print("train plus val must be less than 1.0")
        return 1

    df = pd.read_csv(args.metadata, dtype=str, keep_default_na=False, na_values=[""])
    if df.empty:
        print("metadata.csv has no rows yet, nothing to split")
        return 0

    out = assign(df, args.seed, (args.train, args.val, test))
    out.to_csv(args.metadata, index=False)

    summary = out.groupby("split")["garment_id"].nunique()
    print(f"seed={args.seed} ratios={args.train}/{args.val}/{test}")
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
