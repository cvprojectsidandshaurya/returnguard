#!/usr/bin/env python3
"""Zero shot baseline: embed with a pretrained backbone, match rider shots to packing shots.

This is the reference every later model is judged against. It is deliberately
dumb: no segmentation, no fine tuning, no fusion. Cosine similarity on frozen
features, nothing else.

Query set is rider shots. Gallery is packing shots, grouped by the selected
product or physical-unit identity. The score is the maximum cosine over that
identity's packing shots.

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
from src.fine_tuned_embed import FineTunedEmbedder  # noqa: E402
from src.metrics import recall_at_k, verification  # noqa: E402
from src.training_data import IDENTITY_COLUMNS, identity_column  # noqa: E402

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


def score_matrix(query_vecs: np.ndarray, gallery_vecs: np.ndarray, gallery_identity_idx: np.ndarray, n_identities: int) -> np.ndarray:
    """Queries by configured identities, using max cosine over packing shots."""
    sims = query_vecs @ gallery_vecs.T
    out = np.full((len(query_vecs), n_identities), -np.inf, dtype=np.float32)
    for identity_index in range(n_identities):
        cols = np.where(gallery_identity_idx == identity_index)[0]
        if len(cols):
            out[:, identity_index] = sims[:, cols].max(axis=1)
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
    parser.add_argument("--backbone", default="dinov2_base", help="frozen backbone; ignored when --checkpoint is supplied")
    parser.add_argument("--checkpoint", type=Path, help="fine-tuned checkpoint evaluated with this held-out protocol")
    parser.add_argument("--identity-policy", choices=sorted(IDENTITY_COLUMNS), default="product")
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

    identity_key = identity_column(args.identity_policy)
    if identity_key not in df.columns or df[identity_key].astype(str).str.strip().eq("").any():
        raise ValueError(f"{identity_key} must be populated to evaluate the {args.identity_policy}-identity model")
    identities = sorted(packing[identity_key].unique())
    identity_to_idx = {identity: index for index, identity in enumerate(identities)}

    # Rider shots of an identity with no packing shots in this split cannot be scored.
    rider = rider[rider[identity_key].isin(identity_to_idx)].reset_index(drop=True)
    if rider.empty:
        print(f"no rider shots have a matching gallery {args.identity_policy} identity in this split")
        return 1

    root = Path(args.image_root)
    model_name = f"checkpoint:{args.checkpoint.name}" if args.checkpoint else f"{args.backbone} zero shot"
    print(f"model={model_name} identity_policy={args.identity_policy} split={args.split} identities={len(identities)} "
          f"gallery={len(packing)} queries={len(rider)}")

    embedder = FineTunedEmbedder(args.checkpoint, device=args.device) if args.checkpoint else Embedder(args.backbone, device=args.device)
    print(f"device={embedder.device}, embedding {len(packing) + len(rider)} images")

    gallery_vecs = embedder.encode([root / p for p in packing["relative_path"]], args.batch_size)
    query_vecs = embedder.encode([root / p for p in rider["relative_path"]], args.batch_size)

    gallery_identity_idx = packing[identity_key].map(identity_to_idx).to_numpy()
    correct_index = rider[identity_key].map(identity_to_idx).to_numpy()

    scores = score_matrix(query_vecs, gallery_vecs, gallery_identity_idx, len(identities))
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

    row = (f"| {date.today()} | {commit} | {version} | {model_name} | "
           f"identity={args.identity_policy} split={args.split} seed={args.seed} maxpool | {fmt(overall['tpr_at_1pct_fpr'])} | "
           f"{fmt(overall['recall_at_1'])} | {fmt(overall['recall_at_5'])} | n/a | "
           f"{len(identities)} identities, {overall['n_queries']} queries |")
    print(f"\nrow for docs/results.md:\n{row}")

    if args.no_write:
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model_name = (args.checkpoint.stem if args.checkpoint else args.backbone).replace("/", "-")
    out_path = out_dir / f"{date.today()}-{commit}-{safe_model_name}.md"
    lines = [
        f"# {model_name}, {date.today()}",
        "",
        f"commit `{commit}`, data `{version}`, split `{args.split}`, seed `{args.seed}`",
        "",
        f"Rider shots as queries, packing shots as gallery, max cosine over each {args.identity_policy} identity's packing shots.",
        "No segmentation or score fusion. Fine tuning is used only when --checkpoint is supplied.",
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
