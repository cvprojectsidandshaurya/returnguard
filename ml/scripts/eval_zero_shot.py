#!/usr/bin/env python3
"""Zero shot baseline: embed with a pretrained backbone, match rider shots to packing shots.

This is the reference every later model is judged against. It is deliberately
dumb: no segmentation, no fine tuning, no fusion. Cosine similarity on frozen
features, nothing else.

Query set is rider shots. Gallery is packing shots, grouped per garment. The
score between a rider image and a garment is the max cosine over that garment's
packing shots, which is the multi view rule from CLAUDE.md section 4.5 in its
simplest form.

    python ml/scripts/eval_zero_shot.py --image-root /path/to/images --backbone dinov2_base
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.embed import Embedder  # noqa: E402
from src.metrics import recall_at_k, verification  # noqa: E402

CONDITIONS = {
    "in_polybag": ("true", "false"),
    "lighting": ("dim", "good"),
    "crumpled": ("crumpled", "folded"),
    "tag_visible": ("true", "false"),
    "blurry": ("true", "false"),
    "partial_view": ("true", "false"),
}


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def data_version(metadata_path: Path, df: pd.DataFrame) -> str:
    digest = hashlib.md5(metadata_path.read_bytes()).hexdigest()[:8]
    return f"{df['garment_id'].nunique()}g-{len(df)}img-{digest}"


def score_matrix(query_vecs: np.ndarray, gallery_vecs: np.ndarray, gallery_garment_idx: np.ndarray, n_garments: int) -> np.ndarray:
    """Queries by garments, using max cosine over each garment's packing shots."""
    sims = query_vecs @ gallery_vecs.T
    out = np.full((len(query_vecs), n_garments), -np.inf, dtype=np.float32)
    for g in range(n_garments):
        cols = np.where(gallery_garment_idx == g)[0]
        if len(cols):
            out[:, g] = sims[:, cols].max(axis=1)
    return out


def evaluate(scores: np.ndarray, correct_index: np.ndarray) -> dict:
    result = recall_at_k(scores, correct_index, ks=(1, 5))
    labels = np.zeros_like(scores, dtype=np.int8)
    labels[np.arange(len(correct_index)), correct_index] = 1
    result.update(verification(labels.reshape(-1), scores.reshape(-1)))
    result["n_queries"] = int(len(correct_index))
    return result


def fmt(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--image-root", required=True, help="root of the image store, never inside the repo")
    parser.add_argument("--backbone", default="dinov2_base")
    parser.add_argument("--split", default="test", help="split to evaluate, or 'all' while the dataset is tiny")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out-dir", default="docs/results")
    parser.add_argument("--no-write", action="store_true", help="print only, do not write a report")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    metadata_path = Path(args.metadata)
    df = pd.read_csv(metadata_path, dtype=str, keep_default_na=False, na_values=[""])
    if df.empty:
        print("metadata.csv is empty, shoot some garments first")
        return 1
    if args.split != "all":
        df = df[df["split"] == args.split]
    if df.empty:
        print(f"no rows in split {args.split!r}")
        return 1

    packing = df[df["shot_type"] == "packing"].reset_index(drop=True)
    rider = df[df["shot_type"] == "rider"].reset_index(drop=True)
    if packing.empty or rider.empty:
        print("need both packing and rider shots in the split")
        return 1

    garments = sorted(packing["garment_id"].unique())
    garment_to_idx = {g: i for i, g in enumerate(garments)}

    # Rider shots of a garment with no packing shots in this split cannot be scored.
    rider = rider[rider["garment_id"].isin(garment_to_idx)].reset_index(drop=True)
    if rider.empty:
        print("no rider shots have a matching gallery garment in this split")
        return 1

    root = Path(args.image_root)
    print(f"backbone={args.backbone} split={args.split} garments={len(garments)} "
          f"gallery={len(packing)} queries={len(rider)}")

    embedder = Embedder(args.backbone, device=args.device)
    print(f"device={embedder.device}, embedding {len(packing) + len(rider)} images")

    gallery_vecs = embedder.encode([root / p for p in packing["relative_path"]], args.batch_size)
    query_vecs = embedder.encode([root / p for p in rider["relative_path"]], args.batch_size)

    gallery_garment_idx = packing["garment_id"].map(garment_to_idx).to_numpy()
    correct_index = rider["garment_id"].map(garment_to_idx).to_numpy()

    scores = score_matrix(query_vecs, gallery_vecs, gallery_garment_idx, len(garments))
    overall = evaluate(scores, correct_index)

    breakdown = {}
    for column, values in CONDITIONS.items():
        if column not in rider.columns:
            continue
        for value in values:
            mask = rider[column].astype(str).str.lower() == value
            if mask.sum() < 5:
                continue
            breakdown[f"{column}={value}"] = evaluate(scores[mask.to_numpy()], correct_index[mask.to_numpy()])

    commit = git_commit()
    version = data_version(metadata_path, df)

    print("\noverall")
    print(json.dumps(overall, indent=2))
    print("\nper condition")
    for key, value in breakdown.items():
        print(f"  {key:24s} R@1={fmt(value['recall_at_1'])}  TPR@1%FPR={fmt(value['tpr_at_1pct_fpr'])}  n={value['n_queries']}")

    row = (f"| {date.today()} | {commit} | {version} | {args.backbone} zero shot | "
           f"split={args.split} seed={args.seed} maxpool | {fmt(overall['tpr_at_1pct_fpr'])} | "
           f"{fmt(overall['recall_at_1'])} | {fmt(overall['recall_at_5'])} | n/a | "
           f"{len(garments)} garments, {overall['n_queries']} queries |")
    print(f"\nrow for docs/results.md:\n{row}")

    if args.no_write:
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date.today()}-{commit}-{args.backbone}.md"
    lines = [
        f"# {args.backbone} zero shot, {date.today()}",
        "",
        f"commit `{commit}`, data `{version}`, split `{args.split}`, seed `{args.seed}`",
        "",
        "Rider shots as queries, packing shots as gallery, max cosine over each garment's packing shots.",
        "No segmentation, no fine tuning, no fusion.",
        "",
        "## Overall",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    lines += [f"| {k} | {fmt(v)} |" for k, v in overall.items()]
    lines += ["", "## Per condition", "", "| condition | n queries | R@1 | R@5 | AUC | TPR@1%FPR |", "| --- | --- | --- | --- | --- | --- |"]
    for key, value in breakdown.items():
        lines.append(f"| {key} | {value['n_queries']} | {fmt(value['recall_at_1'])} | "
                     f"{fmt(value['recall_at_5'])} | {fmt(value['auc'])} | {fmt(value['tpr_at_1pct_fpr'])} |")
    lines += ["", "## Row for docs/results.md", "", "```", row, "```", ""]
    out_path.write_text("\n".join(lines))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
