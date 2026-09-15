#!/usr/bin/env python3
"""Run DMS inference on an Edge host and send only results to FastAPI."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backend.app.config import settings  # noqa: E402
from backend.app.services.audio_alert_service import play_vehicle_alert  # noqa: E402
from backend.app.services.inference_service import (  # noqa: E402
    _annotate,
    _event_confidence,
    _metric_sample,
    _public_result,
    _write_evidence_clip,
)
from dms_inference.pipeline import DriverMonitoringPipeline  # noqa: E402


SIMULATED_ROUTE = [
    (10.776889, 106.700806, 24.0, 90.0),
    (10.776975, 106.702103, 27.0, 88.0),
    (10.777126, 106.703411, 31.0, 85.0),
    (10.776402, 106.704096, 22.0, 175.0),
    (10.775514, 106.703901, 18.0, 260.0),
    (10.775365, 106.702551, 26.0, 270.0),
    (10.775803, 106.701201, 20.0, 315.0),
]


class EdgeApi:
    def __init__(self, base_url: str, device_uid: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "X-Edge-Device-ID": device_uid,
            "X-Edge-API-Key": api_key,
        }

    def _request(self, request: urllib.request.Request, timeout: float = 30.0) -> dict:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    body = response.read()
                    return json.loads(body.decode("utf-8")) if body else {}
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")
                if error.code not in {429, 500, 502, 503, 504}:
                    raise RuntimeError(f"API HTTP {error.code}: {detail}") from error
                last_error = RuntimeError(f"API HTTP {error.code}: {detail}")
            except OSError as error:
                last_error = RuntimeError(f"Không kết nối được FastAPI: {error}")
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt))
        assert last_error is not None
        raise last_error

    def post_json(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", **self.headers},
            method="POST",
        )
        return self._request(request)

    def post_file(self, path: str, filename: str, content: bytes, content_type: str) -> dict:
        boundary = f"----dms-edge-{uuid.uuid4().hex}"
        disposition = f'Content-Disposition: form-data; name="file"; filename="{filename}"'
        body = (
            f"--{boundary}\r\n{disposition}\r\nContent-Type: {content_type}\r\n\r\n".encode()
            + content
            + f"\r\n--{boundary}--\r\n".encode()
        )
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}", **self.headers},
            method="POST",
        )
        return self._request(request, timeout=60.0)

    def best_effort_post_json(self, path: str, payload: dict) -> None:
        try:
            self.post_json(path, payload)
        except RuntimeError as error:
            print(f"Cảnh báo kết nối (sẽ tiếp tục xử lý): {error}", file=sys.stderr)

    def best_effort_post_file(self, path: str, filename: str, content: bytes, content_type: str) -> None:
        try:
            self.post_file(path, filename, content, content_type)
        except RuntimeError as error:
            print(f"Không tải được {filename}; Edge vẫn tiếp tục: {error}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Edge Simulator: đọc camera/video, chạy AI cục bộ và gửi kết quả tới FastAPI"
    )
    parser.add_argument("--trip-id", type=int, required=True, help="ID chuyến đang ở trạng thái running")
    parser.add_argument("--source", required=True, help="Đường dẫn video hoặc camera:0")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--yolo-interval", type=int, default=2, choices=range(1, 11), metavar="1..10")
    parser.add_argument("--cnn-interval", type=int, default=2, choices=range(1, 11), metavar="1..10")
    parser.add_argument("--event-cooldown", type=float, default=3.0)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--simulate-gps", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--replace-active-session",
        action="store_true",
        help="Kết thúc phiên mồ côi của chính thiết bị trước khi tạo phiên mới",
    )
    return parser


def resolve_source(value: str) -> tuple[int | Path, str]:
    if value.lower().startswith("camera:"):
        try:
            camera = int(value.split(":", 1)[1])
        except ValueError as error:
            raise SystemExit("Nguồn camera phải có dạng camera:0") from error
        return camera, f"camera:{camera}"
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise SystemExit(f"Không tìm thấy video: {path}")
    return path, path.name


def encode_jpeg(frame: np.ndarray, quality: int = 72) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return encoded.tobytes() if ok else b""


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args()
    if args.event_cooldown < 0:
        raise SystemExit("--event-cooldown phải lớn hơn hoặc bằng 0")

    device_uid = os.getenv("DMS_EDGE_DEVICE_ID", "").strip()
    api_key = os.getenv("DMS_EDGE_API_KEY", "").strip()
    api_base = os.getenv("DMS_API_BASE", "http://127.0.0.1:8000").strip()
    if not device_uid or not api_key:
        raise SystemExit("Thiếu DMS_EDGE_DEVICE_ID hoặc DMS_EDGE_API_KEY")

    source, source_label = resolve_source(args.source)
    for checkpoint in (
        settings.yolo_checkpoint,
        settings.cnn_checkpoint,
        settings.lstm_checkpoint,
        settings.inference_config,
    ):
        if not checkpoint.is_file():
            raise SystemExit(f"Thiếu tệp mô hình/cấu hình: {checkpoint}")

    api = EdgeApi(api_base, device_uid, api_key)
    api.post_json("/api/edge/heartbeat", {"software_version": "edge-simulator-2.0"})
    created = api.post_json("/api/edge/sessions", {
        "trip_id": args.trip_id,
        "source": source_label,
        "device": args.device or "auto",
        "yolo_interval": args.yolo_interval,
        "cnn_interval": args.cnn_interval,
        "save_video": args.save_video,
        "replace_active": args.replace_active_session,
    })
    session_id = int(created["session"]["id"])
    print(f"Edge {device_uid}: bắt đầu phiên #{session_id} từ {source_label}")

    capture = None
    writer = None
    processed = face_frames = 0
    risks: list[int] = []
    metrics: list[dict] = []
    last_event_time: dict[str, float] = {}
    pending_clips: list[dict] = []
    started = time.perf_counter()
    interrupted = False
    error_message: str | None = None
    fps = 25.0
    configured_output = os.getenv("DMS_EDGE_OUTPUT_ROOT", "").strip()
    output_root = Path(configured_output) if configured_output else PROJECT_ROOT / "outputs" / "edge_sessions"
    output_dir = output_root / str(session_id)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(source if isinstance(source, int) else str(source))
        if not capture.isOpened():
            raise RuntimeError(f"Không mở được nguồn {source_label}")
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        evidence_fps = min(max(fps, 1.0), 30.0)
        pre_event_frames: deque[bytes] = deque(maxlen=max(1, int(evidence_fps * 3)))
        clip_target = max(1, int(evidence_fps * 8))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if args.save_video:
            writer = cv2.VideoWriter(
                str(output_dir / "annotated_video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError("Không tạo được video chú thích tại Edge")

        route_index = 0
        last_heartbeat = last_gps = time.monotonic()
        with DriverMonitoringPipeline(
            settings.yolo_checkpoint,
            settings.cnn_checkpoint,
            settings.lstm_checkpoint,
            settings.inference_config,
            device=args.device,
        ) as pipeline:
            while args.max_frames <= 0 or processed < args.max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_started = time.perf_counter()
                frame_index = processed
                timestamp = frame_index / fps
                result = pipeline.process_frame(
                    frame,
                    run_behavior=frame_index % args.yolo_interval == 0,
                    run_face_cnn=frame_index % args.cnn_interval == 0,
                    timestamp_seconds=timestamp,
                )
                processed += 1
                processing_fps = 1.0 / max(time.perf_counter() - frame_started, 1e-6)
                risk = int(result["decision"]["risk_score"])
                risks.append(risk)
                face_frames += int(result["face"] is not None)

                evidence = encode_jpeg(cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA))
                if evidence:
                    pre_event_frames.append(evidence)
                    completed = []
                    for pending in pending_clips:
                        if frame_index > pending["event_frame"]:
                            pending["frames"].append(evidence)
                        if len(pending["frames"]) >= pending["target"]:
                            completed.append(pending)
                    for pending in completed:
                        clip_path = output_dir / f"event_{pending['event_id']}.mp4"
                        if _write_evidence_clip(clip_path, pending["frames"][:pending["target"]], evidence_fps):
                            api.best_effort_post_file(
                                f"/api/edge/sessions/{session_id}/events/{pending['event_id']}/clip",
                                clip_path.name, clip_path.read_bytes(), "video/mp4",
                            )
                        pending_clips.remove(pending)

                rendered = None
                if writer is not None or frame_index % 4 == 0 or result["decision"]["warnings"]:
                    rendered = _annotate(frame, result, timestamp)
                if writer is not None and rendered is not None:
                    writer.write(rendered)

                if frame_index % 4 == 0:
                    metrics.append(_metric_sample(frame_index, timestamp, result, processing_fps))
                if frame_index % 8 == 0:
                    live = _public_result(session_id, frame_index, timestamp, result, processing_fps)
                    api.post_json(
                        f"/api/edge/sessions/{session_id}/metrics",
                        {"samples": metrics, "live_message": live},
                    )
                    metrics.clear()
                    preview = rendered if rendered is not None else _annotate(frame, result, timestamp)
                    preview = cv2.resize(preview, (640, 360), interpolation=cv2.INTER_AREA)
                    preview_jpeg = encode_jpeg(preview, 65)
                    if preview_jpeg:
                        api.best_effort_post_file(
                            f"/api/edge/sessions/{session_id}/preview", "preview.jpg", preview_jpeg, "image/jpeg"
                        )

                created_warning = False
                for warning in result["decision"]["warnings"]:
                    if timestamp - last_event_time.get(warning, -1e9) < args.event_cooldown:
                        continue
                    client_event_id = f"{device_uid}-{session_id}-{frame_index}-{warning}"
                    event_response = api.post_json(f"/api/edge/sessions/{session_id}/events", {
                        "client_event_id": client_event_id,
                        "frame_index": frame_index,
                        "timestamp_seconds": round(timestamp, 3),
                        "event_type": warning,
                        "confidence": round(_event_confidence(warning, result), 6),
                        "risk_score": risk,
                        "payload": {"severity": result["decision"]["severity"], "source": "edge-simulator"},
                    })
                    event_id = int(event_response["event"]["id"])
                    snapshot_frame = rendered if rendered is not None else _annotate(frame, result, timestamp)
                    snapshot = encode_jpeg(snapshot_frame, 88)
                    if snapshot:
                        api.best_effort_post_file(
                            f"/api/edge/sessions/{session_id}/events/{event_id}/snapshot",
                            f"event_{event_id}.jpg", snapshot, "image/jpeg",
                        )
                    pending_clips.append({
                        "event_id": event_id,
                        "event_frame": frame_index,
                        "frames": list(pre_event_frames),
                        "target": clip_target,
                    })
                    last_event_time[warning] = timestamp
                    created_warning = True
                if created_warning:
                    play_vehicle_alert(result["decision"]["severity"], result["decision"]["warnings"])

                now = time.monotonic()
                if now - last_heartbeat >= 20:
                    api.best_effort_post_json("/api/edge/heartbeat", {"software_version": "edge-simulator-2.0"})
                    last_heartbeat = now
                if args.simulate_gps and (processed == 1 or now - last_gps >= 10):
                    lat, lon, speed, heading = SIMULATED_ROUTE[route_index]
                    route_index = (route_index + 1) % len(SIMULATED_ROUTE)
                    api.best_effort_post_json(f"/api/edge/trips/{args.trip_id}/locations", {
                        "latitude": lat, "longitude": lon, "speed_kph": speed,
                        "heading": heading, "source": "gps-simulated",
                    })
                    last_gps = now
    except KeyboardInterrupt:
        interrupted = True
        print("\nĐã nhận Ctrl+C, đang kết thúc phiên Edge...")
    except Exception as error:
        error_message = str(error)
        print(f"Lỗi Edge: {error_message}", file=sys.stderr)
    finally:
        for pending in pending_clips:
            clip_path = output_dir / f"event_{pending['event_id']}.mp4"
            try:
                if _write_evidence_clip(clip_path, pending["frames"], locals().get("evidence_fps", 10.0)):
                    api.best_effort_post_file(
                        f"/api/edge/sessions/{session_id}/events/{pending['event_id']}/clip",
                        clip_path.name, clip_path.read_bytes(), "video/mp4",
                    )
            except Exception as clip_error:
                # Evidence is supplementary. A codec/upload failure must not
                # prevent the session from reporting its final state to FastAPI.
                print(f"Không tạo hoặc tải được clip sự kiện #{pending['event_id']}: {clip_error}", file=sys.stderr)
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        processing_seconds = time.perf_counter() - started
        if metrics and error_message is None:
            try:
                api.post_json(f"/api/edge/sessions/{session_id}/metrics", {"samples": metrics})
            except RuntimeError as upload_error:
                error_message = str(upload_error)
        status = "failed" if error_message else "stopped" if interrupted else "completed"
        summary = {
            "status": status,
            "frame_count": processed,
            "video_duration_seconds": round(processed / max(fps, 1e-6), 3),
            "processing_seconds": round(processing_seconds, 3),
            "avg_fps": round(processed / max(processing_seconds, 1e-6), 3),
            "avg_risk_score": round(float(np.mean(risks)) if risks else 0.0, 3),
            "max_risk_score": max(risks, default=0),
            "face_detection_rate": round(face_frames / max(processed, 1), 4),
            "error_message": error_message,
        }
        try:
            api.post_json(f"/api/edge/sessions/{session_id}/complete", summary)
        except RuntimeError as completion_error:
            print(f"Không thể báo kết thúc phiên cho FastAPI: {completion_error}", file=sys.stderr)
        (output_dir / "session_summary.json").write_text(
            json.dumps({"session_id": session_id, **summary}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({"session_id": session_id, **summary}, ensure_ascii=False, indent=2))

    if error_message:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
