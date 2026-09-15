from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.database import Database


def main() -> None:
    url = os.getenv("DMS_DATABASE_URL", "")
    if not url.startswith(("mysql://", "mysql+mysqlconnector://")):
        raise SystemExit("Hãy đặt DMS_DATABASE_URL trỏ tới MySQL trước khi chạy smoke test.")

    database = Database(url)
    database.initialize()
    suffix = uuid4().hex[:10]
    driver_id: int | None = None
    device_id: int | None = None
    user_id: int | None = None
    session_id: int | None = None
    vehicle_id: int | None = None
    route_id: int | None = None
    trip_id: int | None = None

    try:
        driver = database.create_or_update_driver(
            f"smoke-{suffix}", "Tài xế kiểm thử MySQL", None
        )
        driver_id = int(driver["id"])
        user = database.create_or_update_user(
            f"smoke_{suffix}", "test-only-hash", "Tài khoản kiểm thử",
            "driver", driver_id,
        )
        user_id = int(user["id"])
        vehicle = database.create_vehicle(f"SMK-{suffix[:6]}", "test-car", "MySQL smoke")
        vehicle_id = int(vehicle["id"])
        device = database.create_edge_device(
            f"smoke-edge-{suffix}", "Edge Simulator smoke", "0" * 64, vehicle_id
        )
        device_id = int(device["id"])
        route = database.create_route(
            f"SMK-{suffix}", "Tuyến kiểm thử", "Điểm A", "Điểm B", 12.5
        )
        route_id = int(route["id"])
        trip = database.create_trip(
            f"SMK-TRIP-{suffix}", driver_id, vehicle_id, route_id,
            "2026-09-04T09:00:00+00:00",
        )
        trip_id = int(trip["id"])
        database.update_trip_status(trip_id, "running")
        session_id = database.create_session(
            "edge-simulator:smoke", "cpu", 2, 2, False, trip_id, device_id
        )
        database.update_session(session_id, status="running")
        event_id = database.add_event(session_id, {
            "frame_index": 12, "timestamp_seconds": 0.4, "type": "drowsiness",
            "confidence": 0.91, "risk_score": 70,
        })
        database.add_frame_metrics(session_id, [{
            "frame_index": 12, "timestamp_seconds": 0.4, "ear": 0.17,
            "mar": 0.72, "risk_score": 70, "severity": "high",
            "warnings": "drowsiness", "processing_fps": 18.5,
        }])
        database.add_vehicle_location(
            trip_id, 10.7769, 106.7009, 35.0, 90.0,
            None, "edge-simulator",
        )
        # Repeat the same value to verify MySQL's rowcount=0 update case.
        unchanged = database.set_edge_device_active(device_id, True)
        result = {
            "status": "PASS",
            "database": "mysql",
            "driver_id": driver_id,
            "user_id": user_id,
            "device_id": device_id,
            "session_id": session_id,
            "event_id": event_id,
            "events": len(database.list_events(session_id)),
            "frame_metrics": len(database.list_frame_metrics(session_id)),
            "unchanged_update_returns_record": unchanged is not None,
            "fleet": {
                "vehicle_id": vehicle_id,
                "route_id": route_id,
                "trip_id": trip_id,
                "gps_locations": len(database.list_trip_locations(trip_id)),
            },
        }
        if (
            result["events"] != 1 or result["frame_metrics"] != 1
            or result["fleet"]["gps_locations"] != 1 or unchanged is None
        ):
            raise RuntimeError(f"Smoke test không đạt: {result}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        # Remove only records tagged with this run's random suffix.
        with database.connect() as connection:
            if session_id is not None:
                connection.execute(database._sql("DELETE FROM monitoring_sessions WHERE id = ?"), (session_id,))
            if trip_id is not None:
                connection.execute(database._sql("DELETE FROM trips WHERE id = ?"), (trip_id,))
            if route_id is not None:
                connection.execute(database._sql("DELETE FROM routes WHERE id = ?"), (route_id,))
            if device_id is not None:
                connection.execute(database._sql("DELETE FROM edge_devices WHERE id = ?"), (device_id,))
            if vehicle_id is not None:
                connection.execute(database._sql("DELETE FROM vehicles WHERE id = ?"), (vehicle_id,))
            if user_id is not None:
                connection.execute(database._sql("DELETE FROM app_users WHERE id = ?"), (user_id,))
            if driver_id is not None:
                connection.execute(database._sql("DELETE FROM drivers WHERE id = ?"), (driver_id,))


if __name__ == "__main__":
    main()
