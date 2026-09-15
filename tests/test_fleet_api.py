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


class FleetApiPermissionTests(unittest.TestCase):
    def test_only_supervisor_can_manage_vehicles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Database(Path(temp_dir) / "fleet-api.db")
            database.initialize()
            driver = database.create_or_update_driver("DRV-API", "Tài xế API", None)
            database.create_or_update_user(
                "supervisor_test", hash_password("Supervisor123!"),
                "Supervisor", "supervisor", None,
            )
            database.create_or_update_user(
                "driver_test", hash_password("Driver123!"),
                "Driver", "driver", int(driver["id"]),
            )
            app = FastAPI()
            app.include_router(create_router(database, _ManagerStub()))
            client = TestClient(app)

            supervisor_token = client.post("/api/auth/token", data={
                "username": "supervisor_test", "password": "Supervisor123!",
            }).json()["access_token"]
            driver_token = client.post("/api/auth/token", data={
                "username": "driver_test", "password": "Driver123!",
            }).json()["access_token"]
            supervisor_headers = {"Authorization": f"Bearer {supervisor_token}"}
            driver_headers = {"Authorization": f"Bearer {driver_token}"}

            self.assertEqual(
                client.get("/api/admin/vehicles", headers=driver_headers).status_code, 403
            )
            created = client.post("/api/admin/vehicles", headers=supervisor_headers, json={
                "plate_number": "51A-999.99", "vehicle_type": "car", "model": "Demo",
            })
            self.assertEqual(created.status_code, 201, created.text)
            vehicles = client.get("/api/admin/vehicles", headers=supervisor_headers)
            self.assertEqual(vehicles.status_code, 200)
            self.assertEqual(vehicles.json()["vehicles"][0]["plate_number"], "51A-999.99")

            route = client.post("/api/admin/routes", headers=supervisor_headers, json={
                "route_code": "RT-API", "name": "Kho A đến Kho B",
                "start_location": "Kho A", "end_location": "Kho B",
                "distance_km": 12.5,
            })
            self.assertEqual(route.status_code, 201, route.text)
            trip = client.post("/api/admin/trips", headers=supervisor_headers, json={
                "trip_code": "TRIP-API", "driver_id": int(driver["id"]),
                "vehicle_id": int(created.json()["vehicle"]["id"]),
                "route_id": int(route.json()["route"]["id"]),
            })
            self.assertEqual(trip.status_code, 201, trip.text)
            trip_id = int(trip.json()["trip"]["id"])
            blocked_without_edge = client.patch(
                f"/api/admin/trips/{trip_id}/status", headers=supervisor_headers,
                json={"status": "running"},
            )
            self.assertEqual(blocked_without_edge.status_code, 409)
            self.assertIn("thiết bị Edge", blocked_without_edge.json()["detail"])
            edge = client.post("/api/admin/edge-devices", headers=supervisor_headers, json={
                "device_uid": "edge-fleet-api", "label": "Edge xe kiểm thử",
                "vehicle_id": int(created.json()["vehicle"]["id"]),
            })
            self.assertEqual(edge.status_code, 201, edge.text)
            running = client.patch(
                f"/api/admin/trips/{trip_id}/status", headers=supervisor_headers,
                json={"status": "running"},
            )
            self.assertEqual(running.status_code, 200, running.text)
            self.assertEqual(running.json()["trip"]["status"], "running")
            conflicting_trip = client.post("/api/admin/trips", headers=supervisor_headers, json={
                "trip_code": "TRIP-API-2", "driver_id": int(driver["id"]),
                "vehicle_id": int(created.json()["vehicle"]["id"]),
                "route_id": int(route.json()["route"]["id"]),
            }).json()["trip"]
            conflict = client.patch(
                f"/api/admin/trips/{conflicting_trip['id']}/status", headers=supervisor_headers,
                json={"status": "running"},
            )
            self.assertEqual(conflict.status_code, 409)
            self.assertIn("đang thực hiện chuyến", conflict.json()["detail"])
            completed = client.patch(
                f"/api/admin/trips/{trip_id}/status", headers=supervisor_headers,
                json={"status": "completed"},
            )
            self.assertEqual(completed.status_code, 200, completed.text)
            self.assertEqual(completed.json()["trip"]["status"], "completed")
            self.assertEqual(client.patch(
                f"/api/admin/trips/{trip_id}/status", headers=supervisor_headers,
                json={"status": "running"},
            ).status_code, 409)

            session_id = database.create_session(
                "edge-demo.mp4", "cpu", 2, 2, False, trip_id,
                int(edge.json()["device"]["id"]),
            )
            database.add_vehicle_location(
                trip_id, 10.7769, 106.7009, 35.0, 90.0, None, "simulated",
            )
            removed = client.delete(
                f"/api/admin/trips/{trip_id}/with-data", headers=supervisor_headers,
            )
            self.assertEqual(removed.status_code, 200, removed.text)
            self.assertEqual(removed.json()["deleted_sessions"], 1)
            self.assertIsNone(database.get_trip(trip_id))
            self.assertIsNone(database.get_session(session_id))

            self.assertEqual(
                client.get("/api/admin/assignments", headers=supervisor_headers).status_code,
                404,
            )

    def test_supervisor_can_edit_and_safely_delete_fleet_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Database(Path(temp_dir) / "fleet-crud.db")
            database.initialize()
            driver = database.create_or_update_driver("DRV-CRUD", "Tài xế CRUD", None)
            database.create_or_update_user(
                "supervisor_crud", hash_password("Supervisor123!"),
                "Supervisor", "supervisor", None,
            )
            app = FastAPI()
            app.include_router(create_router(database, _ManagerStub()))
            client = TestClient(app)
            token = client.post("/api/auth/token", data={
                "username": "supervisor_crud", "password": "Supervisor123!",
            }).json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}

            vehicle = client.post("/api/admin/vehicles", headers=headers, json={
                "plate_number": "51A-111.11", "vehicle_type": "car", "model": "A",
            }).json()["vehicle"]
            route = client.post("/api/admin/routes", headers=headers, json={
                "route_code": "RT-CRUD", "name": "Tuyến cũ",
                "start_location": "Kho A", "end_location": "Kho B", "distance_km": 10,
            }).json()["route"]
            trip = client.post("/api/admin/trips", headers=headers, json={
                "trip_code": "TRIP-CRUD", "driver_id": int(driver["id"]),
                "vehicle_id": int(vehicle["id"]), "route_id": int(route["id"]),
            }).json()["trip"]

            updated_vehicle = client.patch(
                f"/api/admin/vehicles/{vehicle['id']}", headers=headers, json={
                    "plate_number": "51A-222.22", "vehicle_type": "SUV", "model": "B",
                    "status": "maintenance", "active": False,
                },
            )
            self.assertEqual(updated_vehicle.status_code, 200, updated_vehicle.text)
            self.assertEqual(updated_vehicle.json()["vehicle"]["plate_number"], "51A-222.22")
            updated_route = client.patch(
                f"/api/admin/routes/{route['id']}", headers=headers, json={
                    "route_code": "RT-CRUD-2", "name": "Tuyến mới",
                    "start_location": "Kho C", "end_location": "Kho D", "distance_km": 12,
                    "active": False,
                },
            )
            self.assertEqual(updated_route.status_code, 200, updated_route.text)
            updated_trip = client.patch(
                f"/api/admin/trips/{trip['id']}", headers=headers, json={
                    "trip_code": "TRIP-CRUD-2", "driver_id": int(driver["id"]),
                    "vehicle_id": int(vehicle["id"]), "route_id": int(route["id"]),
                    "planned_start_at": "2026-09-12T08:00:00Z",
                },
            )
            self.assertEqual(updated_trip.status_code, 200, updated_trip.text)
            self.assertEqual(updated_trip.json()["trip"]["trip_code"], "TRIP-CRUD-2")

            self.assertEqual(client.delete(
                f"/api/admin/vehicles/{vehicle['id']}", headers=headers,
            ).status_code, 409)
            self.assertEqual(client.delete(
                f"/api/admin/routes/{route['id']}", headers=headers,
            ).status_code, 409)
            self.assertEqual(client.delete(
                f"/api/admin/trips/{trip['id']}", headers=headers,
            ).status_code, 200)
            self.assertEqual(client.delete(
                f"/api/admin/routes/{route['id']}", headers=headers,
            ).status_code, 200)
            self.assertEqual(client.delete(
                f"/api/admin/vehicles/{vehicle['id']}", headers=headers,
            ).status_code, 200)

    def test_migration_only_trip_is_not_exposed_as_an_operational_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Database(Path(temp_dir) / "legacy-cleanup.db")
            database.initialize()
            driver = database.create_or_update_driver("DRV-OLD", "Tài xế cũ", None)
            vehicle = database.create_vehicle("LEGACY-UNASSIGNED", "Dữ liệu lưu trữ", None)
            trip = database.create_trip(
                "LEGACY-SESSION-77", int(driver["id"]), int(vehicle["id"]), None, None,
            )
            session_id = database.create_session(
                "legacy.mp4", "cpu", 1, 1, False, int(trip["id"]), None,
            )
            database.update_session(session_id, status="completed")
            database.create_or_update_user(
                "supervisor_cleanup", hash_password("Supervisor123!"),
                "Supervisor", "supervisor", None,
            )
            app = FastAPI()
            app.include_router(create_router(database, _ManagerStub()))
            client = TestClient(app)
            token = client.post("/api/auth/token", data={
                "username": "supervisor_cleanup", "password": "Supervisor123!",
            }).json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}

            response = client.get("/api/admin/trips", headers=headers)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["trips"], [])
            self.assertEqual(client.delete(
                f"/api/admin/trips/{trip['id']}/legacy", headers=headers,
            ).status_code, 404)
            self.assertIsNotNone(database.get_trip(int(trip["id"])))
            self.assertIsNotNone(database.get_session(session_id))


if __name__ == "__main__":
    unittest.main()
