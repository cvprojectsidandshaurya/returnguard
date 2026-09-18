#!/usr/bin/env python3
"""Fit a deployable decision policy from validation data and a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.calibration import SCORE_FIELDS, fit_decision_policy  # noqa: E402
from src.fine_tuned_embed import FineTunedEmbedder  # noqa: E402
from src.set_matching import score_multi_view  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="policy JSON, normally under ignored ml/checkpoints/")
    parser.add_argument("--score-field", default="mean_rider_score", choices=sorted(SCORE_FIELDS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-match-fpr", type=float, default=0.01)
    parser.add_argument("--target-different-false-reject", type=float, default=0.01)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata, dtype=str, keep_default_na=False, na_values=[""])
    val = df[df["split"] == "val"]
    groups = []
    root = Path(args.image_root)
    for garment_id, rows in val.groupby("garment_id"):
        packing = [root / path for path in rows.loc[rows["shot_type"] == "packing", "relative_path"]]
        rider = [root / path for path in rows.loc[rows["shot_type"] == "rider", "relative_path"]]
        if packing and rider:
            groups.append((garment_id, packing, rider))
    if len(groups) < 2:
        raise ValueError("need at least two validation garments with both packing and rider images")

    embedder = FineTunedEmbedder(args.checkpoint, device=args.device)
    vectors = []
    for garment_id, packing, rider in groups:
        vectors.append((garment_id, embedder.encode(packing), embedder.encode(rider)))

    scores, labels = [], []
    for rider_id, _, rider_vectors in vectors:
        for packing_id, packing_vectors, _ in vectors:
            evidence = score_multi_view(packing_vectors, rider_vectors)
            scores.append(getattr(evidence, args.score_field))
            labels.append(rider_id == packing_id)
    policy = fit_decision_policy(
        np.asarray(scores),
        np.asarray(labels),
        score_field=args.score_field,
        target_match_false_positive_rate=args.target_match_fpr,
        target_different_false_reject_rate=args.target_different_false_reject,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    policy.save(args.out)
    print(json.dumps({"policy": policy.as_dict(), "comparisons": len(scores), "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
