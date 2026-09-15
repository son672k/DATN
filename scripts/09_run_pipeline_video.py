#!/usr/bin/env python3
"""Run the complete YOLO + CNN + LSTM pipeline on a video."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dms_inference.pipeline import DriverMonitoringPipeline  # noqa: E402
from dms_training.common import configure_utf8_console  # noqa: E402


COLORS = {
    "normal": (80, 190, 125),
    "warning": (40, 170, 235),
    "danger": (60, 70, 230),
}
WARNING_LABELS = {
    "drowsy": "DROWSY",
    "high_perclos": "HIGH PERCLOS",
    "phone": "PHONE",
    "cigarette": "CIGARETTE",
    "no_seatbelt": "NO SEATBELT",
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run DMS inference on a video.")
    value.add_argument("--video", type=Path, required=True)
    value.add_argument("--yolo", type=Path, default=Path("outputs/final_models/best_yolo.pt"))
    value.add_argument("--cnn", type=Path, default=Path("outputs/final_models/best_cnn.pth"))
    value.add_argument("--lstm", type=Path, default=Path("outputs/final_models/best_lstm.pth"))
    value.add_argument("--config", type=Path, default=Path("configs/inference.json"))
    value.add_argument("--output-dir", type=Path, default=Path("outputs/video_test"))
    value.add_argument("--device", choices=("cpu", "cuda"), default=None)
    value.add_argument("--event-cooldown", type=float, default=3.0)
    value.add_argument("--max-frames", type=int, default=0)
    return value


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def probability(result: dict | None, label: str) -> float | None:
    if not result:
        return None
    return result.get("probabilities", {}).get(label)


def draw_box(frame: np.ndarray, box, label: str, color) -> None:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def annotate(frame: np.ndarray, result: dict, frame_index: int, source_fps: float, processing_fps: float) -> np.ndarray:
    rendered = frame.copy()
    decision = result["decision"]
    color = COLORS[decision["severity"]]
    for detection in result["behavior"]["detections"]:
        draw_box(rendered, detection["box"], f'{detection["class_name"]} {detection["confidence"]:.2f}', (220, 180, 60))
    if result["face"]:
        draw_box(rendered, result["face"]["box"], f'face {result["face"]["confidence"]:.2f}', (90, 215, 155))

    overlay = rendered.copy()
    cv2.rectangle(overlay, (12, 12), (470, 142), (18, 28, 26), -1)
    cv2.addWeighted(overlay, 0.78, rendered, 0.22, 0, rendered)
    cnn_drowsy = probability(result["cnn"], "drowsy")
    lstm_drowsy = probability(result["lstm"], "drowsy")
    lines = [
        (f'Time: {frame_index / source_fps:06.2f}s | Processing: {processing_fps:05.1f} FPS', (225, 225, 225)),
        (f'CNN drowsy: {cnn_drowsy:.3f}' if cnn_drowsy is not None else 'CNN: no face', (220, 220, 220)),
        (f'LSTM drowsy: {lstm_drowsy:.3f}' if lstm_drowsy is not None else f'LSTM buffer: {result["temporal"]["observations"]}/16', (220, 220, 220)),
        (f'Risk: {decision["risk_score"]}/100 | {decision["severity"].upper()}', color),
        ('Warnings: ' + (', '.join(WARNING_LABELS.get(item, item) for item in decision["warnings"]) or 'NONE'), color),
    ]
    for index, (text, line_color) in enumerate(lines):
        cv2.putText(rendered, text, (24, 36 + index * 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, line_color, 1, cv2.LINE_AA)
    return rendered


def main() -> None:
    configure_utf8_console()
    args = parser().parse_args()
    video = absolute(args.video)
    checkpoints = [absolute(args.yolo), absolute(args.cnn), absolute(args.lstm), absolute(args.config)]
    for path in [video, *checkpoints]:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir = absolute(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames > 0:
        total = min(total, args.max_frames)
    video_path = output_dir / "annotated_video.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {video_path}")

    events = []
    event_counts: Counter[str] = Counter()
    last_event_time: dict[str, float] = {}
    risk_values = []
    face_frames = lstm_ready_frames = processed = 0
    started = time.perf_counter()
    jsonl_path = output_dir / "frame_results.jsonl"

    try:
        with DriverMonitoringPipeline(*checkpoints[:3], config_path=checkpoints[3], device=args.device) as pipeline, jsonl_path.open("w", encoding="utf-8") as jsonl:
            progress = tqdm(total=total, desc="DMS video inference", unit="frame")
            while processed < total:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_started = time.perf_counter()
                timestamp = processed / source_fps
                result = pipeline.process_frame(frame, timestamp_seconds=timestamp)
                processed += 1
                elapsed = max(time.perf_counter() - frame_started, 1e-6)
                processing_fps = 1.0 / elapsed
                risk = int(result["decision"]["risk_score"])
                risk_values.append(risk)
                face_frames += int(result["face"] is not None)
                lstm_ready_frames += int(result["lstm"] is not None)

                record = {"frame_index": processed - 1, "timestamp_seconds": round(timestamp, 3), **result}
                jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                for warning in result["decision"]["warnings"]:
                    if timestamp - last_event_time.get(warning, -1e9) >= args.event_cooldown:
                        event = {"type": warning, "frame_index": processed - 1, "timestamp_seconds": round(timestamp, 3), "risk_score": risk}
                        events.append(event)
                        event_counts[warning] += 1
                        last_event_time[warning] = timestamp
                writer.write(annotate(frame, result, processed - 1, source_fps, processing_fps))
                progress.update(1)
            progress.close()
    finally:
        capture.release()
        writer.release()

    wall_seconds = time.perf_counter() - started
    summary = {
        "input_video": str(video),
        "output_video": str(video_path),
        "width": width,
        "height": height,
        "source_fps": source_fps,
        "processed_frames": processed,
        "video_duration_seconds": round(processed / source_fps, 3),
        "processing_seconds": round(wall_seconds, 3),
        "average_processing_fps": round(processed / max(wall_seconds, 1e-6), 3),
        "face_detection_rate": round(face_frames / max(processed, 1), 4),
        "lstm_ready_frames": lstm_ready_frames,
        "average_risk_score": round(float(np.mean(risk_values)) if risk_values else 0.0, 3),
        "max_risk_score": max(risk_values, default=0),
        "event_counts": dict(event_counts),
        "event_count": len(events),
        "device_requested": args.device or "auto",
    }
    (output_dir / "events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "session_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
