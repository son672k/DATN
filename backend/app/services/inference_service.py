from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from ..config import settings
from ..database import Database, utc_now
from .audio_alert_service import play_vehicle_alert

sys.path.insert(0, str(settings.project_root / "src"))
from dms_inference.pipeline import DriverMonitoringPipeline  # noqa: E402


def _event_confidence(warning: str, result: dict) -> float:
    if warning == "drowsy" and result.get("lstm"):
        return float(result["lstm"]["probabilities"].get("drowsy", 0.0))
    if warning == "high_perclos":
        return float(result["perclos"].get("value", 0.0))
    if warning == "yawn" and result.get("geometry"):
        return float(min(1.0, result["geometry"].get("mar", 0.0) / max(result["geometry"].get("mar_threshold", 0.35), 1e-6)))
    if warning == "microsleep" and result.get("geometry"):
        return float(min(1.0, result["geometry"].get("eye_closure_seconds", 0.0) / 3.0))
    if warning == "distraction" and result.get("geometry"):
        return float(min(1.0, result["geometry"].get("distraction_score", 0.0) / 100.0))
    return float(result["behavior"].get("confidence", {}).get(warning, 0.0))


def _draw_box(frame: np.ndarray, box, label: str, color) -> None:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def _annotate(frame: np.ndarray, result: dict, timestamp: float) -> np.ndarray:
    rendered = frame.copy()
    for item in result["behavior"]["detections"]:
        _draw_box(rendered, item["box"], f'{item["class_name"]} {item["confidence"]:.2f}', (220, 180, 60))
    if result["face"]:
        _draw_box(rendered, result["face"]["box"], f'face {result["face"]["confidence"]:.2f}', (90, 215, 155))
    decision = result["decision"]
    color = (60, 70, 230) if decision["severity"] == "danger" else (40, 170, 235) if decision["severity"] == "warning" else (80, 190, 125)
    overlay = rendered.copy()
    cv2.rectangle(overlay, (12, 12), (475, 133), (18, 28, 26), -1)
    cv2.addWeighted(overlay, 0.78, rendered, 0.22, 0, rendered)
    lstm_probability = result["lstm"]["probabilities"].get("drowsy") if result["lstm"] else None
    rows = [
        f"Time: {timestamp:.2f}s",
        f"LSTM drowsy: {lstm_probability:.3f}" if lstm_probability is not None else f'LSTM buffer: {result["temporal"]["observations"]}/16',
        f'EAR: {result["geometry"]["ear"]:.3f} | MAR: {result["geometry"]["mar"]:.3f}' if result.get("geometry") else "EAR / MAR: unavailable",
        f'Risk: {decision["risk_score"]}/100 | ' + (", ".join(decision["warnings"]) or "normal"),
    ]
    for index, label in enumerate(rows):
        cv2.putText(rendered, label, (24, 38 + index * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color if index == 2 else (225, 225, 225), 1, cv2.LINE_AA)
    return rendered


def _write_evidence_clip(path: Path, encoded_frames: list[bytes], fps: float) -> bool:
    if not encoded_frames:
        return False
    first = cv2.imdecode(np.frombuffer(encoded_frames[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    if first is None:
        return False
    height, width = first.shape[:2]
    source_path = path.with_name(f"{path.stem}.source.mp4")
    writer = cv2.VideoWriter(
        str(source_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps), (width, height)
    )
    if not writer.isOpened():
        return False
    try:
        for encoded in encoded_frames:
            frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                writer.write(frame)
    finally:
        writer.release()
    try:
        import imageio_ffmpeg

        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                "-i", str(source_path), "-an", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "25", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (ImportError, OSError, subprocess.CalledProcessError):
        source_path.replace(path)
    else:
        source_path.unlink(missing_ok=True)
    return path.is_file() and path.stat().st_size > 0


def _public_result(session_id: int, frame_index: int, timestamp: float, result: dict, processing_fps: float) -> dict:
    behavior = result["behavior"]
    return {
        "type": "frame",
        "session_id": session_id,
        "frame_index": frame_index,
        "timestamp_seconds": round(timestamp, 3),
        "processing_fps": round(processing_fps, 3),
        "face": result["face"],
        "cnn": result["cnn"],
        "lstm": result["lstm"],
        "temporal": result["temporal"],
        "perclos": result["perclos"],
        "geometry": result.get("geometry"),
        "behavior": {
            "phone": behavior["phone"],
            "cigarette": behavior["cigarette"],
            "seatbelt": behavior["seatbelt"],
            "eye_closed": behavior["eye_closed"],
            "eye_open": behavior["eye_open"],
            "detections": behavior["detections"],
        },
        "decision": result["decision"],
    }


def _metric_sample(frame_index: int, timestamp: float, result: dict, processing_fps: float) -> dict:
    geometry = result.get("geometry") or {}
    cnn = (result.get("cnn") or {}).get("probabilities", {})
    lstm = (result.get("lstm") or {}).get("probabilities", {})
    behavior = result["behavior"]
    decision = result["decision"]
    return {
        "frame_index": frame_index,
        "timestamp_seconds": round(timestamp, 3),
        "ear_left": geometry.get("ear_left"),
        "ear_right": geometry.get("ear_right"),
        "ear": geometry.get("ear"),
        "mar": geometry.get("mar"),
        "geometry_eye_closed": int(bool(geometry.get("eye_closed"))) if geometry else None,
        "yawning": int(bool(geometry.get("yawning"))) if geometry else None,
        "blink_count": int(geometry.get("blink_count", 0)),
        "yawn_count": int(geometry.get("yawn_count", 0)),
        "cnn_awake": cnn.get("awake"),
        "cnn_drowsy": cnn.get("drowsy"),
        "lstm_awake": lstm.get("awake"),
        "lstm_drowsy": lstm.get("drowsy"),
        "perclos": result["perclos"].get("value"),
        "phone": int(bool(behavior["phone"])),
        "cigarette": int(bool(behavior["cigarette"])),
        "seatbelt": int(bool(behavior["seatbelt"])),
        "yolo_eye_closed": int(bool(behavior["eye_closed"])),
        "yolo_eye_open": int(bool(behavior["eye_open"])),
        "risk_score": int(decision["risk_score"]),
        "severity": decision["severity"],
        "warnings": ",".join(decision["warnings"]),
        "processing_fps": round(processing_fps, 3),
        "pitch": geometry.get("pitch"),
        "yaw": geometry.get("yaw"),
        "roll": geometry.get("roll"),
        "gaze_x": geometry.get("gaze_x"),
        "gaze_y": geometry.get("gaze_y"),
        "eye_closure_seconds": geometry.get("eye_closure_seconds", 0.0),
        "max_eye_closure_seconds": geometry.get("max_eye_closure_seconds", 0.0),
        "microsleep": int(bool(geometry.get("microsleep"))),
        "head_distracted": int(bool(geometry.get("head_distracted"))),
        "gaze_distracted": int(bool(geometry.get("gaze_distracted"))),
        "distraction_score": geometry.get("distraction_score", 0.0),
    }


class InferenceManager:
    def __init__(self, database: Database):
        self.database = database
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._active_session_id: int | None = None
        self._subscribers: list[queue.Queue] = []
        self._latest_previews: dict[int, bytes] = {}
        self._latest_preview_session_id: int | None = None

    @property
    def active_session_id(self) -> int | None:
        with self._lock:
            return self._active_session_id

    def subscribe(self) -> queue.Queue:
        # Keep enough ordered messages so a warning is never overwritten by
        # the frame published immediately after it.
        channel: queue.Queue = queue.Queue(maxsize=128)
        with self._lock:
            self._subscribers.append(channel)
        return channel

    def publish_event_review(self, session_id: int, event: dict) -> None:
        """Notify every supervisor dashboard when an event review changes."""
        self._publish({"type": "event_review", "session_id": session_id, "event": event})

    def publish_edge_message(self, message: dict) -> None:
        """Forward authenticated Edge telemetry to existing WebSocket clients."""
        self._publish(message)

    def set_edge_preview(self, session_id: int, jpeg: bytes) -> None:
        self._set_preview(session_id, jpeg)

    def shutdown(self) -> None:
        self.stop()

    def unsubscribe(self, channel: queue.Queue) -> None:
        with self._lock:
            if channel in self._subscribers:
                self._subscribers.remove(channel)

    def latest_preview(self, session_id: int | None = None) -> tuple[int, bytes] | None:
        with self._lock:
            target_id = session_id if session_id is not None else self._latest_preview_session_id
            if target_id is None:
                return None
            jpeg = self._latest_previews.get(target_id)
            return (target_id, jpeg) if jpeg is not None else None

    def _set_preview(self, session_id: int, jpeg: bytes) -> None:
        with self._lock:
            self._latest_previews[session_id] = jpeg
            self._latest_preview_session_id = session_id

    def _publish(self, message: dict) -> None:
        with self._lock:
            channels = list(self._subscribers)
        for channel in channels:
            if channel.full():
                try:
                    channel.get_nowait()
                except queue.Empty:
                    pass
            try:
                channel.put_nowait(message)
            except queue.Full:
                pass

    def resolve_source(self, source: str) -> tuple[str, Path | int]:
        if source.startswith("camera:"):
            raw_index = source.removeprefix("camera:")
            if not raw_index.isdigit() or not 0 <= int(raw_index) <= 9:
                raise ValueError("Nguồn webcam phải có dạng camera:0 đến camera:9")
            camera_index = int(raw_index)
            return f"camera:{camera_index}", camera_index
        candidate = Path(source)
        resolved = (candidate if candidate.is_absolute() else settings.project_root / candidate).resolve()
        if not resolved.is_relative_to(settings.project_root.resolve()):
            raise ValueError("Video phải nằm bên trong thư mục dự án")
        if not resolved.is_file() or resolved.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
            raise FileNotFoundError(resolved)
        return str(resolved), resolved

    def start(
        self,
        source: str,
        device: str | None,
        event_cooldown: float,
        save_video: bool,
        yolo_interval: int,
        cnn_interval: int,
        trip_id: int,
    ) -> int:
        source_label, capture_source = self.resolve_source(source)
        requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if requested_device not in {"cpu", "cuda"}:
            raise ValueError("device phải là cpu hoặc cuda")
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA không khả dụng trên máy này")
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise RuntimeError(f"Phiên {self._active_session_id} đang chạy")
            session_id = self.database.create_session(
                source_label,
                requested_device,
                yolo_interval,
                cnn_interval,
                save_video,
                trip_id,
            )
            self._stop = threading.Event()
            self._active_session_id = session_id
            self._latest_previews.pop(session_id, None)
            self._thread = threading.Thread(
                target=self._run,
                args=(
                    session_id,
                    capture_source,
                    requested_device,
                    event_cooldown,
                    save_video,
                    yolo_interval,
                    cnn_interval,
                ),
                name=f"dms-session-{session_id}",
                daemon=True,
            )
            self._thread.start()
        return session_id

    def stop(self, session_id: int | None = None) -> bool:
        with self._lock:
            if not self._thread or not self._thread.is_alive():
                return False
            if session_id is not None and session_id != self._active_session_id:
                return False
            self._stop.set()
            return True

    def _run(
        self,
        session_id: int,
        source: Path | int,
        device: str,
        event_cooldown: float,
        save_video: bool,
        yolo_interval: int,
        cnn_interval: int,
    ) -> None:
        capture = None
        writer = None
        started = time.perf_counter()
        processed = face_frames = 0
        risk_values: list[int] = []
        metric_buffer: list[dict] = []
        last_event_time: dict[str, float] = {}
        output_dir = settings.session_output_root / str(session_id)
        output_video = output_dir / "annotated_video.mp4"
        snapshots_dir = output_dir / "snapshots"
        clips_dir = output_dir / "clips"
        pending_clips: list[dict] = []
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            snapshots_dir.mkdir(parents=True, exist_ok=True)
            clips_dir.mkdir(parents=True, exist_ok=True)
            self.database.update_session(session_id, status="running", started_at=utc_now())
            capture = cv2.VideoCapture(source if isinstance(source, int) else str(source))
            if not capture.isOpened():
                source_name = f"webcam {source}" if isinstance(source, int) else str(source)
                raise RuntimeError(f"Không mở được nguồn video: {source_name}")
            fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            evidence_fps = min(max(fps, 1.0), 30.0)
            pre_event_frames: deque[bytes] = deque(maxlen=max(1, int(evidence_fps * 3)))
            clip_target_frames = max(1, int(evidence_fps * 8))
            if save_video:
                writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Không tạo được video đầu ra: {output_video}")

            with DriverMonitoringPipeline(
                settings.yolo_checkpoint,
                settings.cnn_checkpoint,
                settings.lstm_checkpoint,
                settings.inference_config,
                device=device,
            ) as pipeline:
                while not self._stop.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frame_started = time.perf_counter()
                    frame_index = processed
                    timestamp = frame_index / fps
                    result = pipeline.process_frame(
                        frame,
                        run_behavior=frame_index % yolo_interval == 0,
                        run_face_cnn=frame_index % cnn_interval == 0,
                        timestamp_seconds=timestamp,
                    )
                    processed += 1
                    frame_fps = 1.0 / max(time.perf_counter() - frame_started, 1e-6)
                    face_frames += int(result["face"] is not None)
                    risk = int(result["decision"]["risk_score"])
                    risk_values.append(risk)
                    evidence_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
                    evidence_ok, evidence_jpeg = cv2.imencode(
                        ".jpg", evidence_frame, [cv2.IMWRITE_JPEG_QUALITY, 72]
                    )
                    evidence_bytes = evidence_jpeg.tobytes() if evidence_ok else b""
                    if evidence_bytes:
                        pre_event_frames.append(evidence_bytes)
                        completed = []
                        for pending in pending_clips:
                            if frame_index > pending["event_frame"]:
                                pending["frames"].append(evidence_bytes)
                            if len(pending["frames"]) >= pending["target"]:
                                _write_evidence_clip(
                                    pending["path"], pending["frames"][:pending["target"]], evidence_fps
                                )
                                completed.append(pending)
                        for pending in completed:
                            pending_clips.remove(pending)
                    if frame_index % 4 == 0:
                        metric_buffer.append(_metric_sample(frame_index, timestamp, result, frame_fps))
                    rendered = None
                    if writer is not None or frame_index % 4 == 0:
                        rendered = _annotate(frame, result, timestamp)
                    if writer is not None and rendered is not None:
                        writer.write(rendered)
                    if frame_index % 4 == 0 and rendered is not None:
                        preview_frame = cv2.resize(
                            rendered,
                            (640, 360),
                            interpolation=cv2.INTER_AREA,
                        )
                        encoded, jpeg = cv2.imencode(
                            ".jpg",
                            preview_frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 65],
                        )
                        if encoded:
                            self._set_preview(session_id, jpeg.tobytes())

                    created_event = False
                    for warning in result["decision"]["warnings"]:
                        if timestamp - last_event_time.get(warning, -1e9) >= event_cooldown:
                            snapshot = snapshots_dir / f"{frame_index:06d}_{warning}.jpg"
                            clip = clips_dir / f"{frame_index:06d}_{warning}.mp4"
                            snapshot_frame = rendered if rendered is not None else _annotate(frame, result, timestamp)
                            cv2.imwrite(str(snapshot), snapshot_frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                            event = {
                                "type": warning,
                                "frame_index": frame_index,
                                "timestamp_seconds": round(timestamp, 3),
                                "confidence": round(_event_confidence(warning, result), 6),
                                "risk_score": risk,
                                "snapshot_path": str(snapshot.relative_to(settings.project_root)),
                                "clip_path": str(clip.relative_to(settings.project_root)),
                            }
                            event["id"] = self.database.add_event(session_id, event)
                            pending_clips.append({
                                "event_frame": frame_index,
                                "frames": list(pre_event_frames),
                                "target": clip_target_frames,
                                "path": clip,
                            })
                            self._publish({"type": "event", "session_id": session_id, "event": event})
                            last_event_time[warning] = timestamp
                            created_event = True
                    if created_event:
                        play_vehicle_alert(result["decision"]["severity"], result["decision"]["warnings"])
                    self._publish(_public_result(session_id, frame_index, timestamp, result, frame_fps))
                    if processed % 24 == 0:
                        self.database.update_session(session_id, frame_count=processed)
                        self.database.add_frame_metrics(session_id, metric_buffer)
                        metric_buffer.clear()

            for pending in pending_clips:
                _write_evidence_clip(pending["path"], pending["frames"], evidence_fps)
            pending_clips.clear()
            self.database.add_frame_metrics(session_id, metric_buffer)
            processing_seconds = time.perf_counter() - started
            status = "stopped" if self._stop.is_set() else "completed"
            summary = {
                "status": status,
                "ended_at": utc_now(),
                "frame_count": processed,
                "video_duration_seconds": round(processed / fps, 3),
                "processing_seconds": round(processing_seconds, 3),
                "avg_fps": round(processed / max(processing_seconds, 1e-6), 3),
                "avg_risk_score": round(float(np.mean(risk_values)) if risk_values else 0.0, 3),
                "max_risk_score": max(risk_values, default=0),
                "face_detection_rate": round(face_frames / max(processed, 1), 4),
                "output_video": str(output_video) if writer is not None else None,
                "yolo_interval": yolo_interval,
                "cnn_interval": cnn_interval,
                "save_video": int(save_video),
            }
            self.database.update_session(session_id, **summary)
            (output_dir / "session_summary.json").write_text(json.dumps({"session_id": session_id, **summary}, ensure_ascii=False, indent=2), encoding="utf-8")
            try:
                from .pdf_report_service import session_pdf, save_atomic
                save_atomic(output_dir / "report.pdf", session_pdf(self.database, session_id, settings.project_root))
            except Exception:
                import logging
                logging.getLogger(__name__).exception("Không tạo được PDF tự động cho phiên %s", session_id)
            self._publish({"type": "session_complete", "session_id": session_id, **summary})
        except Exception as error:
            for pending in pending_clips:
                _write_evidence_clip(pending["path"], pending["frames"], locals().get("evidence_fps", 10.0))
            self.database.add_frame_metrics(session_id, metric_buffer)
            self.database.update_session(session_id, status="failed", ended_at=utc_now(), frame_count=processed, error_message=str(error))
            self._publish({"type": "session_failed", "session_id": session_id, "error": str(error)})
        finally:
            if capture is not None:
                capture.release()
            if writer is not None:
                writer.release()
            with self._lock:
                if self._active_session_id == session_id:
                    self._active_session_id = None
                self._latest_previews.pop(session_id, None)
                if self._latest_preview_session_id == session_id:
                    self._latest_preview_session_id = None
