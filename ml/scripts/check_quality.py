#!/usr/bin/env python3
"""Run capture-quality checks over a metadata split without changing metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.quality import QualityThresholds, assess_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--split", default="all", help="train, val, test, or all")
    parser.add_argument("--shot-type", default="rider", choices=("packing", "rider", "all"))
    parser.add_argument("--min-laplacian-variance", type=float, default=40.0)
    parser.add_argument("--min-brightness", type=float, default=35.0)
    parser.add_argument("--max-brightness", type=float, default=235.0)
    parser.add_argument("--out", help="optional JSON report path, preferably under ignored ml/outputs/")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata, dtype=str, keep_default_na=False, na_values=[""])
    if df.empty:
        print("metadata.csv is empty, nothing to quality check")
        return 1
    if args.split != "all":
        df = df[df["split"] == args.split]
    if args.shot_type != "all":
        df = df[df["shot_type"] == args.shot_type]
    if df.empty:
        print("no photos match the requested split and shot type")
        return 1

    thresholds = QualityThresholds(
        min_brightness=args.min_brightness,
        max_brightness=args.max_brightness,
        min_laplacian_variance=args.min_laplacian_variance,
    )
    records = []
    root = Path(args.image_root)
    for row in df.itertuples(index=False):
        path = root / row.relative_path
        try:
            result = assess_path(path, thresholds).as_dict()
            result.update({"image_id": row.image_id, "relative_path": row.relative_path})
        except Exception as exc:
            result = {"image_id": row.image_id, "relative_path": row.relative_path, "passed": False, "retake_reasons": [f"unreadable_image: {exc}"]}
        records.append(result)

    rejected = [record for record in records if not record["passed"]]
    print(json.dumps({"checked": len(records), "passed": len(records) - len(rejected), "retake": len(rejected), "thresholds": thresholds.__dict__}, indent=2))
    for record in rejected[:20]:
        print(f"RETAKE {record['image_id']}: {', '.join(record['retake_reasons'])}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"thresholds": thresholds.__dict__, "records": records}, indent=2) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
