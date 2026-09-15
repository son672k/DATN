#!/usr/bin/env python3
"""Integration test: FastAPI + WebSocket + pipeline + SQLite."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-interval", type=int, default=2)
    parser.add_argument("--cnn-interval", type=int, default=2)
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()
    with TestClient(app) as client:
        health = client.get("/api/health")
        health.raise_for_status()
        models = client.get("/api/models/status")
        models.raise_for_status()
        if not models.json()["ready"]:
            raise RuntimeError(f"Models are not ready: {models.json()}")

        received_frames = 0
        received_events = []
        with client.websocket_connect("/ws/monitor") as websocket:
            connected = websocket.receive_json()
            if connected["type"] != "connected":
                raise RuntimeError(f"Unexpected WebSocket handshake: {connected}")
            response = client.post(
                "/api/sessions/start",
                json={
                    "source": "data/test_videos/test_video.mp4",
                    "device": "cpu",
                    "event_cooldown_seconds": 3.0,
                    "save_video": args.save_video,
                    "yolo_interval": args.yolo_interval,
                    "cnn_interval": args.cnn_interval,
                },
            )
            response.raise_for_status()
            session_id = response.json()["session_id"]
            while True:
                message = websocket.receive_json()
                if message["type"] == "frame":
                    received_frames += 1
                elif message["type"] == "event":
                    received_events.append(message)
                elif message["type"] == "session_failed":
                    raise RuntimeError(message["error"])
                elif message["type"] == "session_complete":
                    break

        session_response = client.get(f"/api/sessions/{session_id}")
        session_response.raise_for_status()
        event_response = client.get(f"/api/sessions/{session_id}/events")
        event_response.raise_for_status()
        session = session_response.json()["session"]
        stored_events = event_response.json()["events"]
        if session["status"] != "completed" or session["frame_count"] != 240:
            raise RuntimeError(f"Unexpected session result: {session}")
        if not stored_events:
            raise RuntimeError("No events were saved to SQLite")
        summary_response = client.get("/api/metrics/summary")
        summary_response.raise_for_status()
        metric_response = client.get(f"/api/sessions/{session_id}/metrics")
        metric_response.raise_for_status()
        metric_payload = metric_response.json()
        if metric_payload["summary"]["sample_count"] != 60:
            raise RuntimeError(f"Unexpected frame metric samples: {metric_payload['summary']}")
        if not any(sample["ear"] is not None and sample["mar"] is not None for sample in metric_payload["samples"]):
            raise RuntimeError("EAR/MAR were not measured")
        if not any(sample["pitch"] is not None and sample["gaze_x"] is not None for sample in metric_payload["samples"]):
            raise RuntimeError("Head pose/gaze were not measured")
        html_response = client.get(f"/api/sessions/{session_id}/report.html")
        html_response.raise_for_status()
        if "Mất tập trung theo thời gian" not in html_response.text or "sortBy" not in html_response.text:
            raise RuntimeError("Interactive HTML report is incomplete")
        snapshot_events = [event for event in stored_events if event.get("snapshot_path")]
        if not snapshot_events:
            raise RuntimeError("No event snapshots were stored")
        snapshot_response = client.get(f"/api/events/{snapshot_events[0]['id']}/snapshot.jpg")
        snapshot_response.raise_for_status()
        if not snapshot_response.headers.get("content-type", "").startswith("image/jpeg"):
            raise RuntimeError("Snapshot endpoint did not return JPEG")
        csv_response = client.get(f"/api/sessions/{session_id}/report.csv")
        csv_response.raise_for_status()
        if (
            "event_type" not in csv_response.text
            or "session_id" not in csv_response.text
            or not csv_response.text.lstrip("\ufeff").startswith("sep=;\r\n")
        ):
            raise RuntimeError("CSV report is incomplete")
        report = {
            "status": "PASS",
            "session_id": session_id,
            "health": health.json(),
            "processed_frames": session["frame_count"],
            "websocket_frame_messages": received_frames,
            "websocket_event_messages": len(received_events),
            "stored_events": len(stored_events),
            "event_types": [event["event_type"] for event in stored_events],
            "average_processing_fps": session["avg_fps"],
            "face_detection_rate": session["face_detection_rate"],
            "yolo_interval": session["yolo_interval"],
            "cnn_interval": session["cnn_interval"],
            "output_video": session["output_video"],
            "metrics_total_sessions": summary_response.json()["total_sessions"],
            "csv_report_bytes": len(csv_response.content),
            "metric_samples": metric_payload["summary"]["sample_count"],
            "ear_mean": metric_payload["summary"]["ear_mean"],
            "mar_mean": metric_payload["summary"]["mar_mean"],
            "blink_count": metric_payload["summary"]["blink_count"],
            "yawn_count": metric_payload["summary"]["yawn_count"],
            "max_eye_closure_seconds": metric_payload["summary"]["max_eye_closure_seconds"],
            "distraction_score_max": metric_payload["summary"]["distraction_score_max"],
            "html_report_bytes": len(html_response.content),
            "snapshot_events": len(snapshot_events),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
