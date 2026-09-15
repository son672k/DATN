#!/usr/bin/env python3
"""Train the independent five-class behavior YOLO on Habbas11 DMS."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from dms_training.common import configure_utf8_console, write_json  # noqa: E402


def find_data_yaml(dataset_root: Path) -> Path:
    candidates = sorted(dataset_root.rglob("data.yaml")) + sorted(
        dataset_root.rglob("data.yml")
    )
    if not candidates:
        raise FileNotFoundError(f"Không tìm thấy data.yaml trong {dataset_root}")
    if len(candidates) > 1:
        print(f"[WARN] Có nhiều data.yaml; dùng {candidates[0]}")
    return candidates[0]


def build_parser():
    parser = argparse.ArgumentParser(description="Train YOLO 5 lớp Habbas11.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--weights", default="yolov8s.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/yolo"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    configure_utf8_console()
    args = build_parser().parse_args()
    data_yaml = find_data_yaml(args.dataset_root)
    run_root = args.output_dir / "runs"
    model = YOLO(args.weights)
    result = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        project=str(run_root),
        name="habbas11_behavior",
        exist_ok=True,
        pretrained=True,
        plots=True,
    )
    run_dir = Path(result.save_dir)
    best_source = run_dir / "weights" / "best.pt"
    if not best_source.is_file():
        raise FileNotFoundError(f"YOLO không tạo best.pt tại {best_source}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_model = args.output_dir / "best_yolo.pt"
    shutil.copy2(best_source, final_model)

    best_model = YOLO(str(final_model))
    validation = best_model.val(
        data=str(data_yaml), split="val", imgsz=args.image_size, plots=True
    )
    metrics = {
        "data_yaml": str(data_yaml),
        "weights_initialized_from": args.weights,
        "best_model": str(final_model),
        "class_names": best_model.names,
        "precision": float(validation.box.mp),
        "recall": float(validation.box.mr),
        "mAP50": float(validation.box.map50),
        "mAP50_95": float(validation.box.map),
    }
    write_json(args.output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

