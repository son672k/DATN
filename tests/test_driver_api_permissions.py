from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.routes import create_router
from backend.app.database import Database
from backend.app.security import hash_password


class _ManagerStub:
    active_session_id = None


class DriverApiPermissionTests(unittest.TestCase):
    def test_driver_can_only_read_own_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Database(Path(temp_dir) / "driver-api.db")
            database.initialize()
            first = database.create_or_update_driver("DRV-1", "Tài xế Một", None)
            second = database.create_or_update_driver("DRV-2", "Tài xế Hai", None)
            database.create_or_update_user(
                "driver1", hash_password("Driver123!"), "Tài xế Một", "driver", int(first["id"]),
            )
            database.create_or_update_user(
                "driver2", hash_password("Driver123!"), "Tài xế Hai", "driver", int(second["id"]),
            )
            vehicle1 = database.create_vehicle("51A-111.11", "car")
            vehicle2 = database.create_vehicle("51A-222.22", "car")
            trip1 = database.create_trip("TRIP-1", int(first["id"]), int(vehicle1["id"]), None, None)
            trip2 = database.create_trip("TRIP-2", int(second["id"]), int(vehicle2["id"]), None, None)
            session1 = database.create_session("edge:1", "cpu", 2, 2, False, int(trip1["id"]))
            session2 = database.create_session("edge:2", "cpu", 2, 2, False, int(trip2["id"]))
            event1 = database.add_event(session1, {"frame_index": 1, "timestamp_seconds": 0.1, "type": "drowsy", "confidence": 0.9, "risk_score": 80})
            event2 = database.add_event(session2, {"frame_index": 1, "timestamp_seconds": 0.1, "type": "phone", "confidence": 0.9, "risk_score": 70})

            app = FastAPI()
            app.include_router(create_router(database, _ManagerStub()))
            client = TestClient(app)
            token = client.post("/api/auth/token", data={"username": "driver1", "password": "Driver123!"}).json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}

            alerts = client.get("/api/driver/me/alerts", headers=headers)
            self.assertEqual(alerts.status_code, 200)
            self.assertEqual([row["id"] for row in alerts.json()["alerts"]], [event1])
            trips = client.get("/api/driver/me/trips", headers=headers)
            self.assertEqual([row["id"] for row in trips.json()["trips"]], [trip1["id"]])
            self.assertEqual(client.get(f"/api/driver/me/trips/{trip2['id']}/locations", headers=headers).status_code, 404)
            self.assertEqual(client.get(f"/api/events/{event2}/snapshot.jpg", headers=headers).status_code, 403)
            self.assertEqual(client.get("/api/admin/vehicles", headers=headers).status_code, 403)
            self.assertEqual(client.get("/api/admin/analytics", headers=headers).status_code, 403)
            self.assertEqual(client.get("/api/admin/reports/daily.pdf?day=2026-09-10", headers=headers).status_code, 403)
            self.assertEqual(client.get(f"/api/sessions/{session2}/report.pdf", headers=headers).status_code, 403)
            own = client.get(f"/api/driver/me/trips/{trip1['id']}/locations", headers=headers).json()
            self.assertEqual([event["id"] for event in own["events"]], [event1])



if __name__ == "__main__":
    unittest.main()
