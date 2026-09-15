from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.app.database import Database


class EdgeSessionDatabaseTests(unittest.TestCase):
    def test_edge_session_is_linked_to_driver_vehicle_and_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Database(Path(temp_dir) / "edge.db")
            database.initialize()
            driver = database.create_or_update_driver("EDGE-DRV", "Tài xế Edge", None)
            vehicle = database.create_vehicle("51A-EDGE", "car", None)
            route = database.create_route("EDGE-R", "Tuyến Edge", "A", "B", 10)
            trip = database.create_trip(
                "EDGE-TRIP", int(driver["id"]), int(vehicle["id"]),
                int(route["id"]), None,
            )
            database.update_trip_status(int(trip["id"]), "running")
            session_id = database.create_session(
                "camera:0", "cpu", 2, 2, False, int(trip["id"]),
            )
            session = database.get_session(session_id)
            self.assertEqual(session["driver_id"], driver["id"])
            self.assertEqual(session["vehicle_id"], vehicle["id"])
            self.assertEqual(session["trip_id"], trip["id"])
            with database.connect() as connection:
                columns = database._existing_columns(connection, "monitoring_sessions")
            self.assertNotIn("driver_id", columns)
            self.assertNotIn("vehicle_id", columns)
            self.assertEqual(session["plate_number"], "51A-EDGE")
            self.assertEqual(session["trip_code"], "EDGE-TRIP")

    def test_non_mysql_runtime_url_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "MySQL"):
            Database("unsupported://legacy")


if __name__ == "__main__":
    unittest.main()
