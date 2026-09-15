from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import settings  # noqa: E402
from backend.app.database import Database  # noqa: E402


DATETIME_COLUMNS = {
    "drivers": {"created_at": False, "updated_at": False},
    "edge_devices": {"last_seen_at": True, "created_at": False, "updated_at": False},
    "app_users": {"created_at": False, "updated_at": False},
    "vehicles": {"created_at": False, "updated_at": False},
    "routes": {"created_at": False, "updated_at": False},
    "trips": {
        "planned_start_at": True, "started_at": True, "ended_at": True,
        "created_at": False, "updated_at": False,
    },
    "vehicle_locations": {"recorded_at": False, "created_at": False},
    "monitoring_sessions": {"started_at": True, "ended_at": True, "created_at": False},
    "detection_events": {"occurred_at": False, "reviewed_at": True, "gps_recorded_at": True},
}

TYPE_CHANGES = {
    "drivers": {"active": "TINYINT(1) NOT NULL DEFAULT 1"},
    "edge_devices": {
        "active": "TINYINT(1) NOT NULL DEFAULT 1",
        "status": "VARCHAR(20) NOT NULL DEFAULT 'offline'",
    },
    "app_users": {"active": "TINYINT(1) NOT NULL DEFAULT 1"},
    "vehicles": {"active": "TINYINT(1) NOT NULL DEFAULT 1"},
    "routes": {"active": "TINYINT(1) NOT NULL DEFAULT 1", "distance_km": "DOUBLE"},
    "monitoring_sessions": {
        "source": "VARCHAR(500) NOT NULL", "save_video": "TINYINT(1) NOT NULL DEFAULT 1",
        "trip_id": "BIGINT NOT NULL", "edge_device_id": "BIGINT",
        "video_duration_seconds": "DOUBLE NOT NULL DEFAULT 0",
        "processing_seconds": "DOUBLE NOT NULL DEFAULT 0", "avg_fps": "DOUBLE NOT NULL DEFAULT 0",
        "avg_risk_score": "DOUBLE NOT NULL DEFAULT 0",
        "face_detection_rate": "DOUBLE NOT NULL DEFAULT 0",
    },
    "detection_events": {
        "timestamp_seconds": "DOUBLE NOT NULL", "confidence": "DOUBLE NOT NULL",
        "event_payload": "JSON", "severity": "VARCHAR(30)", "latitude": "DOUBLE",
        "longitude": "DOUBLE", "gps_source": "VARCHAR(80)", "reviewed_by": "BIGINT",
    },
    "vehicle_locations": {
        "latitude": "DOUBLE NOT NULL", "longitude": "DOUBLE NOT NULL",
        "speed_kph": "DOUBLE", "heading": "DOUBLE",
    },
    "frame_metrics": {
        **{name: "DOUBLE" for name in (
            "timestamp_seconds", "ear_left", "ear_right", "ear", "mar", "cnn_awake",
            "cnn_drowsy", "lstm_awake", "lstm_drowsy", "perclos", "pitch", "yaw", "roll",
            "gaze_x", "gaze_y",
        )},
        "timestamp_seconds": "DOUBLE NOT NULL",
        "processing_fps": "DOUBLE NOT NULL",
        "eye_closure_seconds": "DOUBLE NOT NULL DEFAULT 0",
        "max_eye_closure_seconds": "DOUBLE NOT NULL DEFAULT 0",
        "distraction_score": "DOUBLE NOT NULL DEFAULT 0",
        **{name: "TINYINT(1) NOT NULL DEFAULT 0" for name in (
            "phone", "cigarette", "seatbelt", "yolo_eye_closed", "yolo_eye_open",
            "microsleep", "head_distracted", "gaze_distracted",
        )},
        "geometry_eye_closed": "TINYINT(1)", "yawning": "TINYINT(1)",
        "warnings": "VARCHAR(500) NOT NULL",
    },
}

INDEXES = {
    "monitoring_sessions": {
        "idx_sessions_trip_id": "trip_id",
        "idx_sessions_edge_device_id": "edge_device_id",
    },
    "detection_events": {"idx_events_occurred_at": "occurred_at"},
    "vehicle_locations": {
        "idx_locations_trip_recorded_at": "trip_id, recorded_at",
    },
}

FOREIGN_KEYS = {
    "edge_devices": {
        "fk_edge_devices_vehicle": "FOREIGN KEY(vehicle_id) REFERENCES vehicles(id) ON DELETE SET NULL",
    },
    "monitoring_sessions": {
        "fk_sessions_trip": "FOREIGN KEY(trip_id) REFERENCES trips(id) ON DELETE RESTRICT",
        "fk_sessions_edge_device": "FOREIGN KEY(edge_device_id) REFERENCES edge_devices(id) ON DELETE SET NULL",
    },
    "detection_events": {
        "fk_events_reviewer": "FOREIGN KEY(reviewed_by) REFERENCES app_users(id) ON DELETE SET NULL",
    },
}


def parse_datetime(value: object) -> datetime | None:
    if value is None or isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed is None:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def column_info(connection, table: str) -> dict[str, str]:
    rows = connection.execute(
        """SELECT COLUMN_NAME AS name, DATA_TYPE AS data_type
        FROM information_schema.columns
        WHERE table_schema = DATABASE() AND table_name = %s""",
        (table,),
    ).fetchall()
    return {str(row["name"]): str(row["data_type"]).lower() for row in rows}


def migrate_datetime(connection, table: str, column: str, nullable: bool) -> None:
    info = column_info(connection, table)
    if column not in info or info[column] in {"datetime", "timestamp"}:
        return
    temporary = f"__dms_{column}_dt"
    if temporary not in info:
        connection.execute(f"ALTER TABLE `{table}` ADD COLUMN `{temporary}` DATETIME(6) NULL")
    rows = connection.execute(
        f"SELECT id, `{column}` AS old_value FROM `{table}` WHERE `{column}` IS NOT NULL"
    ).fetchall()
    for row in rows:
        try:
            converted = parse_datetime(row["old_value"])
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"Không thể đổi {table}.{column} tại id={row['id']}: {row['old_value']}") from error
        connection.execute(
            f"UPDATE `{table}` SET `{temporary}` = %s WHERE id = %s", (converted, row["id"]),
        )
    counts = connection.execute(
        f"""SELECT SUM(`{column}` IS NOT NULL) AS old_count,
        SUM(`{temporary}` IS NOT NULL) AS new_count FROM `{table}`"""
    ).fetchone()
    if int(counts["old_count"] or 0) != int(counts["new_count"] or 0):
        raise RuntimeError(f"Đối chiếu dữ liệu thất bại tại {table}.{column}")
    null_sql = "NULL" if nullable else "NOT NULL"
    connection.execute(
        f"ALTER TABLE `{table}` DROP COLUMN `{column}`, "
        f"CHANGE COLUMN `{temporary}` `{column}` DATETIME(6) {null_sql}"
    )


def constraint_exists(connection, name: str) -> bool:
    row = connection.execute(
        """SELECT COUNT(*) AS total FROM information_schema.table_constraints
        WHERE constraint_schema = DATABASE() AND constraint_name = %s""", (name,),
    ).fetchone()
    return bool(row["total"])


def index_columns(connection, table: str, name: str) -> str | None:
    row = connection.execute(
        """SELECT GROUP_CONCAT(column_name ORDER BY seq_in_index) AS columns_csv
        FROM information_schema.statistics
        WHERE table_schema = DATABASE() AND table_name = %s AND index_name = %s""",
        (table, name),
    ).fetchone()
    return row["columns_csv"] if row else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Tối ưu kiểu dữ liệu và chỉ mục của schema MySQL DMS")
    parser.add_argument("--apply", action="store_true", help="Thực hiện thay đổi sau khi đã sao lưu")
    args = parser.parse_args()
    if not settings.database_url.startswith(("mysql://", "mysql+mysqlconnector://")):
        raise SystemExit("Hãy đặt DMS_DATABASE_URL trỏ tới MySQL trước khi chạy.")
    if not args.apply:
        raise SystemExit("Chạy scripts\\15_backup_mysql.py, sau đó chạy lại với --apply.")

    database = Database(settings.database_url)
    database.initialize()
    with database.connect() as connection:
        for table, columns in DATETIME_COLUMNS.items():
            for column, nullable in columns.items():
                migrate_datetime(connection, table, column, nullable)
        for table, columns in TYPE_CHANGES.items():
            existing = column_info(connection, table)
            for column, definition in columns.items():
                if column in existing:
                    connection.execute(f"ALTER TABLE `{table}` MODIFY COLUMN `{column}` {definition}")
        for table, indexes in INDEXES.items():
            for name, columns in indexes.items():
                actual = index_columns(connection, table, name)
                if not actual:
                    connection.execute(f"CREATE INDEX `{name}` ON `{table}` ({columns})")
        for table, constraints in FOREIGN_KEYS.items():
            for name, definition in constraints.items():
                if not constraint_exists(connection, name):
                    connection.execute(f"ALTER TABLE `{table}` ADD CONSTRAINT `{name}` {definition}")

    print("MySQL schema optimized: DATETIME(6), JSON, TINYINT(1), DOUBLE, indexes and foreign keys.")


if __name__ == "__main__":
    main()
