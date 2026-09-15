from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.routes import create_router
from backend.app.database import Database


class _ManagerStub:
    active_session_id = None

    def __init__(self):
        self.messages = []
        self.previews = {}

    def publish_edge_message(self, message):
        self.messages.append(message)

    def set_edge_preview(self, session_id, jpeg):
        self.previews[session_id] = jpeg


class EdgeIngestionApiTests(unittest.TestCase):
    def test_authenticated_edge_session_metrics_event_and_completion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Database(Path(temp_dir) / "edge-ingestion.db")
            database.initialize()
            driver = database.create_or_update_driver("DRV-INGEST", "Tài xế", None)
            vehicle = database.create_vehicle("51A-345.67", "coach")
            other_vehicle = database.create_vehicle("51A-765.43", "coach")
            trip = database.create_trip(
                "TRIP-INGEST", int(driver["id"]), int(vehicle["id"]), None, None,
            )
            database.update_trip_status(int(trip["id"]), "running")
            from backend.app.security import generate_edge_api_key, hash_edge_api_key
            key = generate_edge_api_key()
            edge = database.create_edge_device("edge-ingest", "Edge cabin", hash_edge_api_key(key), int(vehicle["id"]))
            wrong_key = generate_edge_api_key()
            database.create_edge_device("edge-wrong", "Edge xe khác", hash_edge_api_key(wrong_key), int(other_vehicle["id"]))

            manager = _ManagerStub()
            app = FastAPI()
            app.include_router(create_router(database, manager))
            client = TestClient(app)
            headers = {"X-Edge-Device-ID": "edge-ingest", "X-Edge-API-Key": key}
            wrong_headers = {"X-Edge-Device-ID": "edge-wrong", "X-Edge-API-Key": wrong_key}

            running = client.get("/api/edge/trips/running", headers=headers)
            self.assertEqual(running.status_code, 200, running.text)
            self.assertEqual([item["id"] for item in running.json()["trips"]], [trip["id"]])

            created = client.post("/api/edge/sessions", headers=headers, json={
                "trip_id": int(trip["id"]), "source": "video:cabin-demo.mp4",
                "device": "cpu", "yolo_interval": 2, "cnn_interval": 2,
            })
            self.assertEqual(created.status_code, 201, created.text)
            session = created.json()["session"]
            session_id = int(session["id"])
            self.assertEqual(int(session["edge_device_id"]), int(edge["id"]))
            self.assertEqual(session["status"], "running")
            self.assertEqual(client.post("/api/edge/sessions", headers=headers, json={
                "trip_id": int(trip["id"]), "source": "duplicate",
            }).status_code, 409)

            replacement = client.post("/api/edge/sessions", headers=headers, json={
                "trip_id": int(trip["id"]), "source": "replacement",
                "replace_active": True,
            })
            self.assertEqual(replacement.status_code, 201, replacement.text)
            self.assertEqual(database.get_session(session_id)["status"], "interrupted")
            session = replacement.json()["session"]
            session_id = int(session["id"])
            self.assertEqual(session["status"], "running")

            sample = {
                "frame_index": 0, "timestamp_seconds": 0.0,
                "risk_score": 70, "severity": "danger", "warnings": "phone",
                "processing_fps": 8.5, "phone": 1,
            }
            metrics = client.post(
                f"/api/edge/sessions/{session_id}/metrics", headers=headers,
                json={"samples": [sample], "live_message": {"frame_index": 0}},
            )
            self.assertEqual(metrics.status_code, 200, metrics.text)
            self.assertEqual(len(database.list_frame_metrics(session_id)), 1)
            self.assertEqual(client.post(
                f"/api/edge/sessions/{session_id}/metrics", headers=wrong_headers,
                json={"samples": [sample]},
            ).status_code, 403)

            event_payload = {
                "client_event_id": "edge-ingest-event-0001", "frame_index": 0,
                "timestamp_seconds": 0, "event_type": "phone",
                "confidence": 0.91, "risk_score": 70,
                "payload": {"source": "edge-simulator"},
            }
            first = client.post(
                f"/api/edge/sessions/{session_id}/events", headers=headers, json=event_payload,
            )
            second = client.post(
                f"/api/edge/sessions/{session_id}/events", headers=headers, json=event_payload,
            )
            self.assertEqual(first.status_code, 201, first.text)
            self.assertEqual(second.status_code, 201, second.text)
            self.assertEqual(first.json()["event"]["id"], second.json()["event"]["id"])
            self.assertEqual(len(database.list_events(session_id)), 1)

            completed = client.post(
                f"/api/edge/sessions/{session_id}/complete", headers=headers,
                json={"status": "completed", "frame_count": 1,
                      "video_duration_seconds": 0.04, "processing_seconds": 0.12,
                      "avg_fps": 8.5, "avg_risk_score": 70,
                      "max_risk_score": 70, "face_detection_rate": 1},
            )
            self.assertEqual(completed.status_code, 200, completed.text)
            self.assertEqual(completed.json()["session"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
