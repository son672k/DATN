#!/usr/bin/env python3
"""Crop NTHU faces with the same MediaPipe detector used by inference."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from dms_training.common import configure_utf8_console, write_json  # noqa: E402
from dms_inference.face.mediapipe_face_detector import (  # noqa: E402
    MediaPipeFaceDetector,
)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Manifest rỗng: {path}")
    return rows


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    parser = argparse.ArgumentParser(description="Crop khuôn mặt NTHU bằng MediaPipe.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/processed_nthu")
    )
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser


def main():
    configure_utf8_console()
    args = build_parser().parse_args()
    rows = read_manifest(args.manifest)
    detector = MediaPipeFaceDetector(
        min_confidence=args.min_confidence,
        padding=args.padding,
    )
    faces_root = args.output_root / "faces"
    faces_root.mkdir(parents=True, exist_ok=True)

    successful: list[dict] = []
    failures: list[dict] = []
    for row in tqdm(rows, desc="Crop NTHU faces", ncols=95):
        source = args.dataset_root / row["relative_path"]
        image = cv2.imread(str(source))
        if image is None:
            failures.append({**row, "reason": "image_read_failed"})
            continue
        detection = detector.detect(image)
        if detection is None:
            failures.append({**row, "reason": "face_not_detected"})
            continue
        x1, y1, x2, y2 = detection.box
        crop = detector.crop(image, detection)
        if crop is None:
            failures.append({**row, "reason": "empty_face_crop"})
            continue

        relative_face = (
            Path(row["split"])
            / row["label"]
            / row["subject_id"]
            / f"{row['video_id']}__{int(row['frame_index']):06d}.jpg"
        )
        destination = faces_root / relative_face
        destination.parent.mkdir(parents=True, exist_ok=True)
        saved = cv2.imwrite(
            str(destination), crop, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
        )
        if not saved:
            failures.append({**row, "reason": "image_write_failed"})
            continue
        successful.append(
            {
                **row,
                "face_relative_path": relative_face.as_posix(),
                "face_confidence": f"{detection.confidence:.6f}",
                "face_x1": x1,
                "face_y1": y1,
                "face_x2": x2,
                "face_y2": y2,
            }
        )

    manifest_fields = list(rows[0]) + [
        "face_relative_path",
        "face_confidence",
        "face_x1",
        "face_y1",
        "face_x2",
        "face_y2",
    ]
    write_rows(args.output_root / "processed_manifest.csv", successful, manifest_fields)
    write_rows(
        args.output_root / "crop_failures.csv",
        failures,
        list(rows[0]) + ["reason"],
    )
    summary = {
        "input_frames": len(rows),
        "successful_crops": len(successful),
        "failed_crops": len(failures),
        "success_rate": len(successful) / max(1, len(rows)),
        "by_split": dict(sorted(Counter(row["split"] for row in successful).items())),
        "by_label": dict(sorted(Counter(row["label"] for row in successful).items())),
        "face_detector": "MediaPipe Face Detection",
        "min_confidence": args.min_confidence,
        "padding": args.padding,
    }
    write_json(args.output_root / "crop_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["success_rate"] < 0.90:
        print("[WARN] Tỷ lệ crop dưới 90%; hãy kiểm tra ngưỡng và failures CSV.")
    detector.close()


if __name__ == "__main__":
    main()
