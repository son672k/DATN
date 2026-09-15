#!/usr/bin/env python3
"""Pretrain EfficientNet-B0 on the face-cropped DDD image dataset."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from dms_training.cnn import (  # noqa: E402
    ImageRecordDataset,
    build_efficientnet_b0,
    eval_transform,
    train_cnn,
    train_transform,
)
from dms_training.common import (  # noqa: E402
    LABEL_TO_ID,
    choose_device,
    configure_utf8_console,
    seed_everything,
    write_json,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def infer_ddd_label(path: Path, root: Path) -> str | None:
    for component in reversed(path.relative_to(root).parts[:-1]):
        name = component.strip().lower().replace("_", " ").replace("-", " ")
        if "non" in name or "not" in name or "awake" in name or "alert" in name:
            return "awake"
        if "drow" in name or "sleep" in name:
            return "drowsy"
    return None


def collect_ddd(root: Path) -> list[dict]:
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = infer_ddd_label(path, root)
        if label is not None:
            records.append(
                {"path": path, "label": label, "label_id": LABEL_TO_ID[label]}
            )
    counts = Counter(record["label"] for record in records)
    if set(counts) != {"awake", "drowsy"}:
        raise RuntimeError(f"DDD không có đủ hai lớp: {counts}")
    print(f"DDD records: {len(records)}; phân bố: {dict(counts)}")
    return records


def stratified_split(records: list[dict], val_ratio: float, seed: int):
    train_records = []
    val_records = []
    rng = random.Random(seed)
    for label_id in (0, 1):
        rows = [row for row in records if row["label_id"] == label_id]
        rng.shuffle(rows)
        n_val = max(1, round(len(rows) * val_ratio))
        val_records.extend(rows[:n_val])
        train_records.extend(rows[n_val:])
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def build_parser():
    parser = argparse.ArgumentParser(description="Pretrain EfficientNet-B0 bằng DDD.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/cnn_pretrain")
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    configure_utf8_console()
    args = build_parser().parse_args()
    seed_everything(args.seed)
    device = choose_device()
    records = collect_ddd(args.dataset_root)
    train_records, val_records = stratified_split(
        records, args.val_ratio, args.seed
    )
    train_loader = DataLoader(
        ImageRecordDataset(train_records, train_transform()),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        ImageRecordDataset(val_records, eval_transform()),
        batch_size=args.batch * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = build_efficientnet_b0(pretrained=True)
    checkpoint_path = args.output_dir / "cnn_pretrained_ddd.pth"
    _, history = train_cnn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_path=checkpoint_path,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience_limit=args.patience,
        checkpoint_metadata={"training_stage": "DDD pretraining"},
        train_records=train_records,
    )
    write_json(args.output_dir / "history.json", history)
    summary = {
        "device": str(device),
        "train_images": len(train_records),
        "val_images": len(val_records),
        "checkpoint": str(checkpoint_path),
        "note": "DDD split chỉ dùng pretrain; đánh giá cuối phải dùng NTHU subject-wise.",
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

