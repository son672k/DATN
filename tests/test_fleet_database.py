from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.app.database import Database


class FleetDatabaseTests(unittest.TestCase):
    def test_vehicle_route_trip_and_gps_flow(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Database(Path(temp_dir) / "fleet.db")
            database.initialize()
            driver = database.create_or_update_driver("DRV-001", "Tài xế A", None)
            vehicle = database.create_vehicle("51a-123.45", "car", "Demo 2026")
            route = database.create_route(
                "SG-BD", "Sài Gòn - Bình Dương", "TP.HCM", "Bình Dương", 32.5
            )
            trip = database.create_trip(
                "TRIP-001", int(driver["id"]), int(vehicle["id"]),
                int(route["id"]), "2026-09-04T09:00:00+00:00",
            )

            self.assertEqual(vehicle["plate_number"], "51A-123.45")
            running = database.update_trip_status(int(trip["id"]), "running")
            self.assertEqual(running["status"], "running")
            self.assertIsNotNone(running["started_at"])

            location = database.add_vehicle_location(
                int(trip["id"]), 10.7769, 106.7009,
                42.5, 90.0, None, "edge-simulator",
            )
            self.assertAlmostEqual(float(location["latitude"]), 10.7769)
            self.assertEqual(len(database.list_trip_locations(int(trip["id"]))), 1)
            self.assertEqual(database.get_trip(int(trip["id"]))["plate_number"], "51A-123.45")

            completed = database.update_trip_status(int(trip["id"]), "completed")
            self.assertEqual(completed["status"], "completed")


if __name__ == "__main__":
    unittest.main()
