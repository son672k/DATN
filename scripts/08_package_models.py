#!/usr/bin/env python3
"""Verify and package the final YOLO, CNN and LSTM checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from dms_training.common import configure_utf8_console, write_json  # noqa: E402


def load_torch_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_parser():
    parser = argparse.ArgumentParser(description="Đóng gói ba model cuối cùng.")
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--cnn", type=Path, required=True)
    parser.add_argument("--lstm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final_models"))
    return parser


def main():
    configure_utf8_console()
    args = build_parser().parse_args()
    for path in (args.yolo, args.cnn, args.lstm):
        if not path.is_file():
            raise FileNotFoundError(path)

    yolo = YOLO(str(args.yolo))
    yolo_names = yolo.names
    normalized_names = {int(key): str(value) for key, value in yolo_names.items()}
    expected = {"closed_eye", "open_eye", "cigarette", "phone", "seatbelt"}
    actual = {name.lower().replace(" ", "_") for name in normalized_names.values()}
    if not expected.issubset(actual):
        raise RuntimeError(f"YOLO thiếu lớp; hiện có {normalized_names}")

    cnn = load_torch_checkpoint(args.cnn)
    if cnn.get("model_name") != "efficientnet_b0" or cnn.get("feature_dim") != 1280:
        raise RuntimeError("CNN checkpoint không phải EfficientNet-B0 1280-D mong đợi")
    lstm = load_torch_checkpoint(args.lstm)
    if lstm.get("input_dim") != 1280 or lstm.get("seq_len") != 16:
        raise RuntimeError("LSTM checkpoint không có input_dim=1280 và seq_len=16")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.yolo, args.output_dir / "best_yolo.pt")
    shutil.copy2(args.cnn, args.output_dir / "best_cnn.pth")
    shutil.copy2(args.lstm, args.output_dir / "best_lstm.pth")
    packaged = ["best_yolo.pt", "best_cnn.pth", "best_lstm.pth"]
    manifest = {
        "files": packaged,
        "face_detector": {
            "name": "MediaPipe Face Detection",
            "packaged_checkpoint": False,
            "purpose": "face ROI only; not counted as a trained project model",
        },
        "yolo_classes": normalized_names,
        "cnn": {
            "model_name": cnn.get("model_name"),
            "feature_dim": cnn.get("feature_dim"),
            "class_names": cnn.get("class_names"),
            "split": cnn.get("split"),
        },
        "lstm": {
            "input_dim": lstm.get("input_dim"),
            "hidden_dim": lstm.get("hidden_dim"),
            "num_layers": lstm.get("num_layers"),
            "seq_len": lstm.get("seq_len"),
            "class_names": lstm.get("class_names"),
            "split": lstm.get("split"),
        },
    }
    write_json(args.output_dir / "model_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
