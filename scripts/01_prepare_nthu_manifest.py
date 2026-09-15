#!/usr/bin/env python3
"""Build a leakage-safe frame manifest for the NTHU drowsiness dataset.

This step deliberately performs no image augmentation, cropping, or model
training. It establishes the single source of truth that CNN and LSTM stages
must reuse so a subject never leaks across train/validation/test splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_DATASET_CANDIDATES = (
    Path("/kaggle/input/datasets/samymesbah/nthu-dataset-ddd-multi-class"),
    Path("/kaggle/input/nthu-dataset-ddd-multi-class"),
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_TO_ID = {"awake": 0, "drowsy": 1}


def configure_utf8_console() -> None:
    """Keep Vietnamese CLI messages readable on Windows code pages."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (AttributeError, OSError):
                pass


@dataclass(frozen=True)
class FrameRecord:
    relative_path: str
    subject_id: str
    video_id: str
    frame_index: int
    scenario: str
    raw_label: str
    label: str
    label_id: int
    split: str = ""


@dataclass(frozen=True)
class SkippedFile:
    relative_path: str
    reason: str


def normalize_label(raw_label: str) -> str | None:
    """Map exact NTHU labels without the `drowsy` substring trap."""
    normalized = raw_label.strip().lower().replace("-", "").replace("_", "")
    if normalized in {"notdrowsy", "nondrowsy", "awake", "alert"}:
        return "awake"
    if normalized in {"drowsy", "sleepy"}:
        return "drowsy"
    return None


def infer_label_from_path(path: Path) -> tuple[str, str] | None:
    """Fall back to exact directory components when the filename lacks a label."""
    for component in reversed(path.parts[:-1]):
        label = normalize_label(component)
        if label is not None:
            return component.lower(), label
    return None


def parse_nthu_path(path: Path, dataset_root: Path) -> tuple[FrameRecord | None, str | None]:
    """Parse names such as `005_glasses_yawning_162_notdrowsy.jpg`."""
    try:
        relative_path = path.relative_to(dataset_root).as_posix()
    except ValueError:
        relative_path = path.as_posix()

    parts = path.stem.split("_")
    if len(parts) < 3:
        return None, "filename_has_too_few_parts"

    subject_id = parts[0]
    if not re.fullmatch(r"\d{3}", subject_id):
        return None, "subject_id_is_not_three_digits"

    raw_label = parts[-1].lower()
    label = normalize_label(raw_label)
    frame_position = len(parts) - 2

    if label is None:
        inferred = infer_label_from_path(path)
        if inferred is None:
            return None, "label_not_found"
        raw_label, label = inferred
        frame_position = len(parts) - 1

    if frame_position < 1:
        return None, "frame_index_not_found"
    try:
        frame_index = int(parts[frame_position])
    except ValueError:
        return None, "frame_index_is_not_integer"

    scenario_parts = parts[1:frame_position]
    scenario = "_".join(scenario_parts) if scenario_parts else "unspecified"

    # Label remains part of video_id because this Kaggle derivative stores
    # awake/drowsy tracks separately. It prevents a temporal window from
    # crossing an artificial label boundary.
    video_id = "_".join((subject_id, scenario, raw_label))

    return (
        FrameRecord(
            relative_path=relative_path,
            subject_id=subject_id,
            video_id=video_id,
            frame_index=frame_index,
            scenario=scenario,
            raw_label=raw_label,
            label=label,
            label_id=LABEL_TO_ID[label],
        ),
        None,
    )


def find_dataset_root(explicit_root: Path | None) -> Path:
    candidates = (explicit_root,) if explicit_root is not None else DEFAULT_DATASET_CANDIDATES
    for candidate in candidates:
        if candidate is not None and candidate.is_dir():
            return candidate.resolve()
    checked = "\n".join(f"  - {path}" for path in candidates if path is not None)
    raise FileNotFoundError(
        "Không tìm thấy NTHU dataset. Hãy truyền --dataset-root. Đã kiểm tra:\n"
        f"{checked}"
    )


def scan_dataset(dataset_root: Path) -> tuple[list[FrameRecord], list[SkippedFile]]:
    records: list[FrameRecord] = []
    skipped: list[SkippedFile] = []
    paths = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise RuntimeError(f"Không tìm thấy ảnh trong {dataset_root}")

    for path in paths:
        record, reason = parse_nthu_path(path, dataset_root)
        if record is None:
            skipped.append(
                SkippedFile(
                    relative_path=path.relative_to(dataset_root).as_posix(),
                    reason=reason or "unknown_parse_error",
                )
            )
        else:
            records.append(record)

    if not records:
        raise RuntimeError("Không có file NTHU nào phân tích thành công")
    return records, skipped


def assign_subject_splits(
    subject_ids: Iterable[str],
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, str]:
    subjects = sorted(set(subject_ids))
    if len(subjects) < 3:
        raise ValueError(
            "Cần ít nhất 3 subject để tạo train/val/test subject-wise; "
            f"chỉ tìm thấy {subjects}"
        )
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("train_ratio và val_ratio phải nằm trong khoảng (0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio phải nhỏ hơn 1")

    rng = random.Random(seed)
    rng.shuffle(subjects)
    total = len(subjects)
    n_train = max(1, round(total * train_ratio))
    n_val = max(1, round(total * val_ratio))
    if n_train + n_val >= total:
        n_train = total - 2
        n_val = 1

    assignment: dict[str, str] = {}
    for index, subject_id in enumerate(subjects):
        if index < n_train:
            split = "train"
        elif index < n_train + n_val:
            split = "val"
        else:
            split = "test"
        assignment[subject_id] = split
    return assignment


def apply_splits(
    records: Iterable[FrameRecord], assignment: dict[str, str]
) -> list[FrameRecord]:
    result = [
        FrameRecord(**{**asdict(record), "split": assignment[record.subject_id]})
        for record in records
    ]
    return sorted(
        result,
        key=lambda row: (row.split, row.subject_id, row.video_id, row.frame_index),
    )


def audit_manifest(records: list[FrameRecord]) -> dict:
    if not records:
        raise ValueError("Manifest rỗng")

    paths = [record.relative_path for record in records]
    duplicates = [path for path, count in Counter(paths).items() if count > 1]
    if duplicates:
        raise RuntimeError(f"Phát hiện đường dẫn trùng trong manifest: {duplicates[:5]}")

    split_subjects: dict[str, set[str]] = defaultdict(set)
    for record in records:
        split_subjects[record.split].add(record.subject_id)

    required_splits = {"train", "val", "test"}
    if set(split_subjects) != required_splits:
        raise RuntimeError(f"Thiếu split: hiện có {sorted(split_subjects)}")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_subjects[left] & split_subjects[right]
        if overlap:
            raise RuntimeError(f"Subject leakage giữa {left}/{right}: {sorted(overlap)}")

    summary: dict[str, object] = {
        "status": "PASS - subject sets are disjoint",
        "total_frames": len(records),
        "total_subjects": len({record.subject_id for record in records}),
        "total_videos": len({record.video_id for record in records}),
        "labels": dict(sorted(Counter(record.label for record in records).items())),
        "splits": {},
    }
    split_summary: dict[str, object] = {}
    for split in ("train", "val", "test"):
        rows = [record for record in records if record.split == split]
        split_summary[split] = {
            "subjects": sorted(split_subjects[split]),
            "num_subjects": len(split_subjects[split]),
            "num_videos": len({record.video_id for record in rows}),
            "num_frames": len(rows),
            "labels": dict(sorted(Counter(record.label for record in rows).items())),
        }
    summary["splits"] = split_summary
    return summary


def write_csv(path: Path, rows: Iterable[object], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def save_outputs(
    output_dir: Path,
    dataset_root: Path,
    records: list[FrameRecord],
    skipped: list[SkippedFile],
    assignment: dict[str, str],
    summary: dict,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "nthu_manifest.csv",
        records,
        list(FrameRecord.__dataclass_fields__),
    )
    write_csv(
        output_dir / "skipped_files.csv",
        skipped,
        list(SkippedFile.__dataclass_fields__),
    )
    (output_dir / "subject_split.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "dataset_root_at_creation": str(dataset_root),
                "assignment": dict(sorted(assignment.items())),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    summary_with_context = {
        **summary,
        "dataset_root_at_creation": str(dataset_root),
        "skipped_files": len(skipped),
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary_with_context, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tạo manifest NTHU và chia train/val/test theo tài xế."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Thư mục gốc NTHU. Nếu bỏ qua, script tự dò đường dẫn Kaggle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/manifests"),
        help="Thư mục lưu CSV/JSON (mặc định: outputs/manifests).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser


def main() -> None:
    configure_utf8_console()
    args = build_argument_parser().parse_args()
    dataset_root = find_dataset_root(args.dataset_root)
    records, skipped = scan_dataset(dataset_root)
    assignment = assign_subject_splits(
        (record.subject_id for record in records),
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    split_records = apply_splits(records, assignment)
    summary = audit_manifest(split_records)
    save_outputs(
        output_dir=args.output_dir,
        dataset_root=dataset_root,
        records=split_records,
        skipped=skipped,
        assignment=assignment,
        summary=summary,
        seed=args.seed,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nĐã lưu manifest tại: {args.output_dir.resolve()}")
    if skipped:
        print(f"Cảnh báo: {len(skipped)} file không phân tích được; xem skipped_files.csv")


if __name__ == "__main__":
    main()
