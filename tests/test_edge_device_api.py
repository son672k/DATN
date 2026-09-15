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


class EdgeDeviceApiTests(unittest.TestCase):
    def test_edge_key_heartbeat_and_vehicle_scope(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Database(Path(temp_dir) / "edge.db")
            database.initialize()
            database.create_or_update_user(
                "supervisor", hash_password("Supervisor123!"),
                "Supervisor", "supervisor", None,
            )
            driver = database.create_or_update_driver("DRV-EDGE", "Tài xế Edge", None)
            vehicle = database.create_vehicle("51A-123.45", "car")
            other_vehicle = database.create_vehicle("51A-543.21", "car")
            trip = database.create_trip(
                "TRIP-EDGE", int(driver["id"]), int(vehicle["id"]), None, None,
            )
            database.update_trip_status(int(trip["id"]), "running")

            app = FastAPI()
            app.include_router(create_router(database, _ManagerStub()))
            client = TestClient(app)
            token = client.post("/api/auth/token", data={
                "username": "supervisor", "password": "Supervisor123!",
            }).json()["access_token"]
            admin_headers = {"Authorization": f"Bearer {token}"}
            created = client.post("/api/admin/edge-devices", headers=admin_headers, json={
                "device_uid": "edge-001", "label": "Cabin Edge",
                "vehicle_id": int(vehicle["id"]),
            })
            self.assertEqual(created.status_code, 201, created.text)
            api_key = created.json()["api_key"]
            edge_headers = {"X-Edge-Device-ID": "edge-001", "X-Edge-API-Key": api_key}
            self.assertEqual(client.post("/api/edge/heartbeat", headers=edge_headers, json={}).status_code, 200)
            self.assertEqual(client.post("/api/edge/heartbeat", headers={
                **edge_headers, "X-Edge-API-Key": "wrong",
            }, json={}).status_code, 401)
            gps = client.post(
                f"/api/edge/trips/{trip['id']}/locations", headers=edge_headers,
                json={"latitude": 10.77, "longitude": 106.70},
            )
            self.assertEqual(gps.status_code, 201, gps.text)

            reassigned = client.patch(
                f"/api/admin/edge-devices/{created.json()['device']['id']}",
                headers=admin_headers,
                json={"label": "Cabin Edge cập nhật", "vehicle_id": int(other_vehicle["id"])},
            )
            self.assertEqual(reassigned.status_code, 200, reassigned.text)
            self.assertEqual(reassigned.json()["device"]["vehicle_id"], int(other_vehicle["id"]))
            self.assertEqual(reassigned.json()["device"]["status"], "offline")
            self.assertEqual(client.post(
                f"/api/edge/trips/{trip['id']}/locations", headers=edge_headers,
                json={"latitude": 10.77, "longitude": 106.70},
            ).status_code, 409)

            restored = client.patch(
                f"/api/admin/edge-devices/{created.json()['device']['id']}",
                headers=admin_headers,
                json={"label": "Cabin Edge", "vehicle_id": int(vehicle["id"])},
            )
            self.assertEqual(restored.status_code, 200, restored.text)

            wrong = client.post("/api/admin/edge-devices", headers=admin_headers, json={
                "device_uid": "edge-002", "label": "Wrong Edge",
                "vehicle_id": int(other_vehicle["id"]),
            }).json()
            wrong_headers = {"X-Edge-Device-ID": "edge-002", "X-Edge-API-Key": wrong["api_key"]}
            self.assertEqual(client.post(
                f"/api/edge/trips/{trip['id']}/locations", headers=wrong_headers,
                json={"latitude": 10.77, "longitude": 106.70},
            ).status_code, 409)

            rotated = client.post(
                f"/api/admin/edge-devices/{created.json()['device']['id']}/rotate-key",
                headers=admin_headers,
            )
            self.assertEqual(rotated.status_code, 200, rotated.text)
            self.assertEqual(
                client.post("/api/edge/heartbeat", headers=edge_headers, json={}).status_code,
                401,
            )
            new_headers = {
                "X-Edge-Device-ID": "edge-001",
                "X-Edge-API-Key": rotated.json()["api_key"],
            }
            self.assertEqual(
                client.post("/api/edge/heartbeat", headers=new_headers, json={}).status_code,
                200,
            )
            wrong_id = int(wrong["device"]["id"])
            self.assertEqual(client.delete(
                f"/api/admin/edge-devices/{wrong_id}", headers=admin_headers,
            ).status_code, 409)
            revoked = client.patch(
                f"/api/admin/edge-devices/{wrong_id}/active", headers=admin_headers,
                json={"active": False},
            )
            self.assertEqual(revoked.status_code, 200, revoked.text)
            self.assertEqual(client.delete(
                f"/api/admin/edge-devices/{wrong_id}", headers=admin_headers,
            ).status_code, 200)


if __name__ == "__main__":
    unittest.main()
