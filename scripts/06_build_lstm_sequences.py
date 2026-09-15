#!/usr/bin/env python3
"""Extract CNN features once and build leakage-safe temporal index windows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from dms_training.cnn import (  # noqa: E402
    ImageRecordDataset,
    eval_transform,
    extract_features,
    load_cnn_checkpoint,
)
from dms_training.common import (  # noqa: E402
    choose_device,
    configure_utf8_console,
    seed_everything,
    write_json,
)


def read_records(manifest: Path, face_root: Path):
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {"train": [], "val": [], "test": []}
    for row in rows:
        split = row.get("split")
        if split not in result:
            continue
        path = face_root / row["face_relative_path"]
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu ảnh: {path}")
        result[split].append(
            {
                **row,
                "path": path,
                "label_id": int(row["label_id"]),
                "frame_index": int(row["frame_index"]),
            }
        )
    for split in result:
        result[split].sort(
            key=lambda row: (row["subject_id"], row["video_id"], row["frame_index"])
        )
        if not result[split]:
            raise RuntimeError(f"Split {split} rỗng")
    return result


def write_feature_manifest(path: Path, records: list[dict]):
    fields = [
        "feature_index",
        "subject_id",
        "video_id",
        "frame_index",
        "label",
        "label_id",
        "split",
        "face_relative_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(records):
            writer.writerow(
                {field: index if field == "feature_index" else record[field] for field in fields}
            )


@torch.no_grad()
def extract_split_features(model, records, batch, workers, device):
    dataset = ImageRecordDataset(records, eval_transform())
    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    chunks = []
    model.eval()
    for images, _, _ in tqdm(loader, desc="Extract CNN features", ncols=90):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            feature = extract_features(model, images)
        chunks.append(feature.float().cpu().numpy())
    features = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if features.shape != (len(records), 1280):
        raise RuntimeError(f"Kích thước feature không mong đợi: {features.shape}")
    return features


def build_windows(records, sequence_length, stride, max_frame_gap):
    groups = defaultdict(list)
    for feature_index, record in enumerate(records):
        key = (record["subject_id"], record["video_id"], record["label_id"])
        groups[key].append((feature_index, record))

    windows = []
    for (subject_id, video_id, label_id), items in sorted(groups.items()):
        items.sort(key=lambda item: item[1]["frame_index"])
        for start in range(0, len(items) - sequence_length + 1, stride):
            window = items[start : start + sequence_length]
            frame_indices = [item[1]["frame_index"] for item in window]
            if any(
                right - left <= 0 or right - left > max_frame_gap
                for left, right in zip(frame_indices, frame_indices[1:])
            ):
                continue
            windows.append(
                {
                    "indices": [item[0] for item in window],
                    "label": int(label_id),
                    "subject_id": subject_id,
                    "video_id": video_id,
                    "start_frame": frame_indices[0],
                    "end_frame": frame_indices[-1],
                }
            )
    if not windows:
        raise RuntimeError("Không tạo được chuỗi; kiểm tra tên frame và MAX_FRAME_GAP")
    return windows


def save_windows(path: Path, windows: list[dict]):
    np.savez_compressed(
        path,
        indices=np.asarray([row["indices"] for row in windows], dtype=np.int64),
        labels=np.asarray([row["label"] for row in windows], dtype=np.int64),
        subject_ids=np.asarray([row["subject_id"] for row in windows]),
        video_ids=np.asarray([row["video_id"] for row in windows]),
        start_frames=np.asarray([row["start_frame"] for row in windows], dtype=np.int64),
        end_frames=np.asarray([row["end_frame"] for row in windows], dtype=np.int64),
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Tạo feature và chuỗi cho LSTM.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--face-root", type=Path, required=True)
    parser.add_argument("--cnn-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/lstm_data")
    )
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-frame-gap", type=int, default=5)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    configure_utf8_console()
    args = build_parser().parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device()
    model, checkpoint = load_cnn_checkpoint(args.cnn_checkpoint, device)
    split_records = read_records(args.manifest, args.face_root)
    summary = {
        "cnn_checkpoint": str(args.cnn_checkpoint),
        "cnn_epoch": checkpoint.get("epoch"),
        "sequence_length": args.sequence_length,
        "stride": args.stride,
        "max_frame_gap": args.max_frame_gap,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        print(f"\n===== {split.upper()} =====")
        records = split_records[split]
        features = extract_split_features(
            model, records, args.batch, args.workers, device
        )
        feature_path = args.output_dir / f"features_{split}.npy"
        np.save(feature_path, features, allow_pickle=False)
        write_feature_manifest(
            args.output_dir / f"feature_manifest_{split}.csv", records
        )
        windows = build_windows(
            records, args.sequence_length, args.stride, args.max_frame_gap
        )
        save_windows(args.output_dir / f"sequences_{split}.npz", windows)
        labels = Counter(row["label"] for row in windows)
        summary["splits"][split] = {
            "frames": len(records),
            "features_shape": list(features.shape),
            "sequences": len(windows),
            "sequence_labels": {"awake": labels[0], "drowsy": labels[1]},
            "subjects": sorted({row["subject_id"] for row in records}),
        }
    write_json(args.output_dir / "sequence_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

