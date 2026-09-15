from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import Dataset
from torchvision import transforms

from .common import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD


def build_efficientnet_b0(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    try:
        model = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0
        )
    except Exception as exc:
        if not pretrained:
            raise
        print(f"[WARN] Không tải được ImageNet weights ({exc}); khởi tạo ngẫu nhiên.")
        model = timm.create_model(
            "efficientnet_b0", pretrained=False, num_classes=0
        )
    feature_dim = model.num_features
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(feature_dim),
        nn.Dropout(0.40),
        nn.Linear(feature_dim, 256),
        nn.SiLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.30),
        nn.Linear(256, num_classes),
    )
    return model


def train_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class ImageRecordDataset(Dataset):
    """Dataset over dictionaries containing `path` and integer `label_id`."""

    def __init__(self, records: list[dict], transform):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record["path"]) as image:
            image = image.convert("RGB")
        return self.transform(image), int(record["label_id"]), index


def class_weights(records: Iterable[dict], device: torch.device) -> torch.Tensor:
    counts = Counter(int(record["label_id"]) for record in records)
    if set(counts) != {0, 1}:
        raise ValueError(f"Cần đủ hai lớp awake/drowsy, hiện có: {counts}")
    total = sum(counts.values())
    weights = [total / (len(counts) * counts[index]) for index in range(2)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def amp_context(device: torch.device):
    return torch.amp.autocast(
        device_type=device.type, enabled=device.type == "cuda"
    )


@torch.no_grad()
def evaluate_classifier(model, loader, device, criterion=None) -> dict:
    model.eval()
    losses: list[float] = []
    truth: list[int] = []
    predictions: list[int] = []
    for images, labels, *_ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with amp_context(device):
            logits = model(images)
            if criterion is not None:
                losses.append(float(criterion(logits, labels).item()) * len(labels))
        truth.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    if not truth:
        raise RuntimeError("DataLoader đánh giá rỗng")
    return {
        "loss": sum(losses) / len(truth) if losses else 0.0,
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


def train_cnn(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    checkpoint_path: Path,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience_limit: int,
    checkpoint_metadata: dict,
    train_records: list[dict],
) -> tuple[nn.Module, dict]:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_records, device), label_smoothing=0.05
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_loss = float("inf")
    best_state = None
    patience = 0

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        seen = 0
        for images, labels, *_ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * len(labels)
            correct += int((logits.argmax(1) == labels).sum().item())
            seen += len(labels)
        scheduler.step()

        train_loss = running_loss / max(1, seen)
        train_acc = correct / max(1, seen)
        val_metrics = evaluate_classifier(model, val_loader, device, criterion)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        print(
            f"Epoch {epoch:02d}/{epochs}: train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            patience = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state": best_state,
                    "model_name": "efficientnet_b0",
                    "class_names": CLASS_NAMES,
                    "num_classes": 2,
                    "feature_dim": 1280,
                    "epoch": epoch,
                    "val_loss": best_loss,
                    **checkpoint_metadata,
                },
                checkpoint_path,
            )
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"Early stopping sau {epoch} epoch")
                break

    if best_state is None:
        raise RuntimeError("CNN không tạo được checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def load_cnn_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model = build_efficientnet_b0(
        num_classes=int(checkpoint.get("num_classes", 2)), pretrained=False
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def extract_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    feature_map = model.forward_features(images)
    return model.forward_head(feature_map, pre_logits=True)

