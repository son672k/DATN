#!/usr/bin/env python3
"""Train the final subject-wise LSTM over frozen EfficientNet-B0 features."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from dms_training.common import (  # noqa: E402
    CLASS_NAMES,
    choose_device,
    configure_utf8_console,
    seed_everything,
    write_json,
)
from dms_training.lstm import FeatureSequenceDataset, SubjectWiseLSTMClassifier  # noqa: E402


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    truth = []
    predictions = []
    for sequences, labels in loader:
        sequences = sequences.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(sequences)
            loss = criterion(logits, labels)
        losses.append(float(loss.item()) * len(labels))
        truth.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    return {
        "loss": sum(losses) / max(1, len(truth)),
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision_macro": float(
            precision_score(truth, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(truth, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(truth, predictions, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            truth,
            predictions,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Train LSTM subject-wise.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lstm"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    configure_utf8_console()
    args = build_parser().parse_args()
    seed_everything(args.seed)
    device = choose_device()
    datasets = {
        split: FeatureSequenceDataset(
            args.data_dir / f"features_{split}.npy",
            args.data_dir / f"sequences_{split}.npz",
        )
        for split in ("train", "val", "test")
    }
    feature_dim = int(datasets["train"].features.shape[1])
    sequence_length = int(datasets["train"].indices.shape[1])
    train_counts = Counter(datasets["train"].labels.tolist())
    if set(train_counts) != {0, 1}:
        raise RuntimeError(f"Train sequences không đủ hai lớp: {train_counts}")
    sample_weights = [1.0 / train_counts[int(label)] for label in datasets["train"].labels]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.batch * 2,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.batch * 2,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        ),
    }
    model = SubjectWiseLSTMClassifier(
        input_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    # WeightedRandomSampler already balances the training batches. Applying
    # inverse-frequency weights again in the loss would correct the same
    # moderate class imbalance twice and could bias the minority class.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience = 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_lstm.pth"
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        seen = 0
        for sequences, labels in loaders["train"]:
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                logits = model(sequences)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * len(labels)
            correct += int((logits.argmax(1) == labels).sum().item())
            seen += len(labels)
        scheduler.step()
        train_loss = running_loss / max(1, seen)
        train_acc = correct / max(1, seen)
        val_metrics = evaluate(model, loaders["val"], criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        print(
            f"Epoch {epoch:02d}/{args.epochs}: train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            patience = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state": best_state,
                    "class_names": CLASS_NAMES,
                    "input_dim": feature_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "num_classes": 2,
                    "seq_len": sequence_length,
                    "dropout": args.dropout,
                    "epoch": epoch,
                    "val_loss": best_loss,
                    "split": "subject-wise NTHU",
                    "model_type": "SubjectWiseLSTMClassifier",
                },
                checkpoint_path,
            )
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping sau {epoch} epoch")
                break
    if best_state is None:
        raise RuntimeError("LSTM không tạo được checkpoint")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders["test"], criterion, device)
    write_json(args.output_dir / "history.json", history)
    write_json(args.output_dir / "test_metrics.json", test_metrics)
    summary = {
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "input_dim": feature_dim,
        "sequence_length": sequence_length,
        "train_sequences": len(datasets["train"]),
        "val_sequences": len(datasets["val"]),
        "test_sequences": len(datasets["test"]),
        "test_metrics": test_metrics,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
