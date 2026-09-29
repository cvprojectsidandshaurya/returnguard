#!/usr/bin/env python3
"""Fit a unit-identity decision policy from validation data and a checkpoint."""

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


def bootstrap_summary(
    scores: np.ndarray,
    labels: np.ndarray,
    score_field: str,
    target_match_fpr: float,
    target_different_false_reject: float,
    samples: int,
    seed: int,
) -> dict:
    """Estimate calibration uncertainty without mixing validation into test data."""
    if samples < 1:
        return {}
    rng = np.random.default_rng(seed)
    positives = scores[labels]
    negatives = scores[~labels]
    rows = []
    for _ in range(samples):
        draw_scores = np.concatenate(
            [rng.choice(positives, size=len(positives), replace=True), rng.choice(negatives, size=len(negatives), replace=True)]
        )
        draw_labels = np.array([True] * len(positives) + [False] * len(negatives))
        policy = fit_decision_policy(
            draw_scores,
            draw_labels,
            score_field=score_field,
            target_match_false_positive_rate=target_match_fpr,
            target_different_false_reject_rate=target_different_false_reject,
        )
        rows.append(
            {
                "different_unit_threshold": policy.different_unit_threshold,
                "match_threshold": policy.match_threshold,
                "match_tpr": policy.observed_match_true_positive_rate,
                "different_tnr": policy.observed_different_true_negative_rate,
            }
        )
    return {
        key: {"p05": float(np.quantile([row[key] for row in rows], 0.05)), "p95": float(np.quantile([row[key] for row in rows], 0.95))}
        for key in rows[0]
    }


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
    parser.add_argument("--min-positive-examples", type=int, default=40)
    parser.add_argument("--min-negative-examples", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata, dtype=str, keep_default_na=False, na_values=[""])
    val = df[df["split"] == "val"]
    groups = []
    root = Path(args.image_root)
    required = {"unit_id", "design_id"}
    missing = sorted(required - set(val.columns))
    if missing:
        raise ValueError(f"metadata is missing required columns for unit calibration: {missing}")
    for unit_id, rows in val.groupby("unit_id"):
        if not str(unit_id).strip():
            raise ValueError("unit_id must be populated for unit-identity calibration")
        packing = [root / path for path in rows.loc[rows["shot_type"] == "packing", "relative_path"]]
        rider = [root / path for path in rows.loc[rows["shot_type"] == "rider", "relative_path"]]
        if packing and rider:
            groups.append((unit_id, packing, rider))
    if len(groups) < 2:
        raise ValueError("need at least two validation garments with both packing and rider images")

    embedder = FineTunedEmbedder(args.checkpoint, device=args.device)
    vectors = []
    for unit_id, packing, rider in groups:
        vectors.append((unit_id, embedder.encode(packing), embedder.encode(rider)))

    scores, labels = [], []
    for rider_unit_id, _, rider_vectors in vectors:
        for packing_unit_id, packing_vectors, _ in vectors:
            evidence = score_multi_view(packing_vectors, rider_vectors)
            scores.append(getattr(evidence, args.score_field))
            labels.append(rider_unit_id == packing_unit_id)
    score_values = np.asarray(scores)
    same_unit = np.asarray(labels)
    policy = fit_decision_policy(
        score_values,
        same_unit,
        score_field=args.score_field,
        target_match_false_positive_rate=args.target_match_fpr,
        target_different_false_reject_rate=args.target_different_false_reject,
        min_positive_examples=args.min_positive_examples,
        min_negative_examples=args.min_negative_examples,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    policy.save(args.out)
    report = {
        "policy": policy.as_dict(),
        "comparisons": len(scores),
        "positive_comparisons": int(same_unit.sum()),
        "negative_comparisons": int((~same_unit).sum()),
        "comparison_protocol": "all validation unit-to-unit packing/rider set comparisons",
        "bootstrap_90pct_interval": bootstrap_summary(
            score_values,
            same_unit,
            args.score_field,
            args.target_match_fpr,
            args.target_different_false_reject,
            args.bootstrap_samples,
            args.bootstrap_seed,
        ),
        "out": str(args.out),
    }
    report_path = args.out.with_suffix(args.out.suffix + ".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report, "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
