#!/usr/bin/env python3
"""Check data/metadata.csv for the mistakes that quietly invalidate results.

Run this before any evaluation. Exits non zero on any error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.splits import split_groups  # noqa: E402

REQUIRED = [
    "image_id", "garment_id", "design_id", "unit_id", "category", "color",
    "shot_type", "relative_path", "phone_id", "lighting", "in_polybag",
    "crumpled", "tag_visible", "blurry", "partial_view", "split",
    "protocol_version",
]

ALLOWED = {
    "shot_type": {"packing", "rider"},
    "lighting": {"good", "dim", "mixed"},
    "crumpled": {"flat", "folded", "crumpled"},
    "split": {"train", "val", "test"},
    "category": {"kurta", "tee", "shirt", "jeans", "dress"},
}

BOOLS = ["in_polybag", "tag_visible", "blurry", "partial_view"]


def validate(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []

    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        return [f"missing required columns: {missing_cols}"]

    if df.empty:
        return errors

    for col in REQUIRED:
        blank = df[df[col].isna() | (df[col].astype(str).str.strip() == "")]
        if not blank.empty:
            errors.append(f"{col}: {len(blank)} blank values, first at image_id={blank.iloc[0]['image_id']}")

    dupes = df[df.duplicated("image_id", keep=False)]
    if not dupes.empty:
        errors.append(f"duplicate image_id: {sorted(dupes['image_id'].unique())[:5]}")

    for col, allowed in ALLOWED.items():
        bad = sorted(set(df[col].dropna().astype(str)) - allowed)
        if bad:
            errors.append(f"{col}: unexpected values {bad}")

    for col in BOOLS:
        bad = sorted(set(df[col].dropna().astype(str).str.lower()) - {"true", "false"})
        if bad:
            errors.append(f"{col}: not boolean, found {bad}")

    # A garment must sit in exactly one split.
    per_garment = df.groupby("garment_id")["split"].nunique()
    straddlers = per_garment[per_garment > 1]
    if not straddlers.empty:
        errors.append(f"garments in more than one split: {list(straddlers.index)[:5]}")

    # A split group must sit in exactly one split. This is the leakage that matters.
    groups = split_groups(df)
    tagged = df.assign(split_group=df["garment_id"].map(groups))
    per_group = tagged.groupby("split_group")["split"].nunique()
    leaks = per_group[per_group > 1]
    if not leaks.empty:
        for root in list(leaks.index)[:5]:
            members = sorted(tagged[tagged["split_group"] == root]["garment_id"].unique())
            errors.append(f"split group {root} leaks across splits, garments: {members}")

    # Every garment needs both sides of the pair to be usable.
    counts = df.pivot_table(index="garment_id", columns="shot_type", values="image_id", aggfunc="count").fillna(0)
    for side in ("packing", "rider"):
        if side not in counts.columns:
            errors.append(f"no {side} shots anywhere in the dataset")
            continue
        short = counts[counts[side] < 3]
        if not short.empty:
            errors.append(f"garments with fewer than 3 {side} shots: {list(short.index)[:5]}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/metadata.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata, dtype=str, keep_default_na=False, na_values=[""])
    errors = validate(df)

    if errors:
        print(f"FAIL: {len(errors)} problem(s) in {args.metadata}")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"OK: {len(df)} rows, {df['garment_id'].nunique() if not df.empty else 0} garments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
