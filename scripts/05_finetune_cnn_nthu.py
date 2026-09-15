#!/usr/bin/env python3
"""Fine-tune and evaluate EfficientNet-B0 on subject-wise NTHU face crops."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from dms_training.cnn import (  # noqa: E402
    ImageRecordDataset,
    build_efficientnet_b0,
    eval_transform,
    evaluate_classifier,
    train_cnn,
    train_transform,
)
from dms_training.common import (  # noqa: E402
    choose_device,
    configure_utf8_console,
    seed_everything,
    write_json,
)


def load_processed_records(manifest: Path, face_root: Path) -> dict[str, list[dict]]:
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    split_records = {"train": [], "val": [], "test": []}
    for row in rows:
        split = row["split"]
        if split not in split_records:
            continue
        path = face_root / row["face_relative_path"]
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu face crop: {path}")
        split_records[split].append(
            {
                **row,
                "path": path,
                "label_id": int(row["label_id"]),
            }
        )
    for split, records in split_records.items():
        counts = Counter(record["label"] for record in records)
        if set(counts) != {"awake", "drowsy"}:
            raise RuntimeError(f"Split {split} không đủ hai lớp: {counts}")
        print(f"{split}: {len(records)} frames; {dict(counts)}")
    return split_records


def load_initial_weights(model, checkpoint_path: Path, device):
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    print(f"Đã nạp pretrained CNN: {checkpoint_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Fine-tune CNN bằng NTHU.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--face-root", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cnn_nthu"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    configure_utf8_console()
    args = build_parser().parse_args()
    seed_everything(args.seed)
    device = choose_device()
    split_records = load_processed_records(args.manifest, args.face_root)
    loaders = {
        "train": DataLoader(
            ImageRecordDataset(split_records["train"], train_transform()),
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
        ),
        "val": DataLoader(
            ImageRecordDataset(split_records["val"], eval_transform()),
            batch_size=args.batch * 2,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        ),
        "test": DataLoader(
            ImageRecordDataset(split_records["test"], eval_transform()),
            batch_size=args.batch * 2,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        ),
    }
    model = build_efficientnet_b0(pretrained=args.pretrained_checkpoint is None)
    if args.pretrained_checkpoint is not None:
        load_initial_weights(model, args.pretrained_checkpoint, device)
    checkpoint_path = args.output_dir / "best_cnn.pth"
    model, history = train_cnn(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=device,
        checkpoint_path=checkpoint_path,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience_limit=args.patience,
        checkpoint_metadata={
            "training_stage": "NTHU subject-wise fine-tuning",
            "split": "subject-wise NTHU",
        },
        train_records=split_records["train"],
    )
    test_metrics = evaluate_classifier(
        model,
        loaders["test"],
        device,
        criterion=nn.CrossEntropyLoss(),
    )
    write_json(args.output_dir / "history.json", history)
    write_json(args.output_dir / "test_metrics.json", test_metrics)
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))
    print(f"CNN cuối cùng: {checkpoint_path}")


if __name__ == "__main__":
    main()
