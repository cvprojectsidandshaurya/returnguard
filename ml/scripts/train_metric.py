#!/usr/bin/env python3
"""Fine tune DINOv2 on packing-to-rider garment pairs.

This trains a retrieval embedding only. It does not choose decision thresholds
or report held-out metrics. Calibration and test reporting remain separate from
model development and use the validation/test tooling owned by the data role.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.losses import symmetric_infonce  # noqa: E402
from src.metric_model import DinoMetricModel  # noqa: E402
from src.training_data import (  # noqa: E402
    CrossDomainPairDataset,
    HardNegativeBatchSampler,
    assert_image_paths_exist,
    load_metadata,
)


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_collate(processor, training: bool):
    augment = transforms.Compose(
        [
            transforms.RandomResizedCrop(518, scale=(0.65, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.12, hue=0.03),
        ]
    )

    def load(path: Path) -> Image.Image:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        return augment(rgb) if training else rgb

    def collate(batch):
        packing, rider, garment_ids = zip(*batch)
        packing_values = processor(images=[load(path) for path in packing], return_tensors="pt")["pixel_values"]
        rider_values = processor(images=[load(path) for path in rider], return_tensors="pt")["pixel_values"]
        return packing_values, rider_values, garment_ids

    return collate


@torch.no_grad()
def validation_loss(model, loader, device: str, temperature: float) -> float:
    model.eval()
    losses: list[float] = []
    for packing, rider, _ in loader:
        loss = symmetric_infonce(model(packing.to(device)), model(rider.to(device)), temperature)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def metadata_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--image-root", required=True, help="image store root, never a directory inside git")
    parser.add_argument("--backbone", default="dinov2_base", choices=("dinov2_base", "dinov2_large"))
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--train-last-blocks", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-dir", default="ml/checkpoints")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 2 or args.lr <= 0:
        parser.error("epochs must be positive, batch-size must be at least 2, and lr must be positive")

    seed_everything(args.seed)
    metadata_path = Path(args.metadata)
    df = load_metadata(metadata_path)
    image_root = Path(args.image_root)
    train_data = CrossDomainPairDataset(df, image_root, split="train", seed=args.seed)
    val_data = CrossDomainPairDataset(df, image_root, split="val", seed=args.seed)
    if len(train_data) < args.batch_size or len(val_data) < 2:
        raise ValueError("need at least batch-size train garments and two validation garments with both shot types")
    assert_image_paths_exist(train_data)
    assert_image_paths_exist(val_data)

    device = pick_device(args.device)
    model = DinoMetricModel(args.backbone, args.projection_dim, args.train_last_blocks).to(device)
    train_sampler = HardNegativeBatchSampler(train_data, args.batch_size, args.seed, drop_last=True)
    val_sampler = HardNegativeBatchSampler(val_data, args.batch_size, args.seed, drop_last=False)
    train_loader = DataLoader(train_data, batch_sampler=train_sampler, num_workers=0, collate_fn=make_collate(model.processor, True))
    val_loader = DataLoader(val_data, batch_sampler=val_sampler, num_workers=0, collate_fn=make_collate(model.processor, False))
    optimizer = AdamW(model.trainable_parameter_groups(args.lr), weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    run = {
        **vars(args),
        "device": device,
        "metadata_sha256_12": metadata_digest(metadata_path),
        "train_garments": len(train_data),
        "val_garments": len(val_data),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(run, sort_keys=True))

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_data.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        losses: list[float] = []
        for packing, rider, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = symmetric_infonce(model(packing.to(device)), model(rider.to(device)), args.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        val = validation_loss(model, val_loader, device, args.temperature)
        scheduler.step()
        result = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": val}
        print(json.dumps(result, sort_keys=True))
        payload = {"model": model.state_dict(), "run": run, "epoch": epoch + 1, "val_loss": val}
        torch.save(payload, save_dir / "last.pt")
        if val < best_val:
            best_val = val
            torch.save(payload, save_dir / "best.pt")

    print(f"saved best checkpoint to {save_dir / 'best.pt'} (validation loss {best_val:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
