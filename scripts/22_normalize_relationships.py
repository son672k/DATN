from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.database import Database  # noqa: E402


def columns(connection, table: str) -> set[str]:
    rows = connection.execute(
        """SELECT COLUMN_NAME AS name FROM information_schema.columns
        WHERE table_schema=DATABASE() AND table_name=%s""",
        (table,),
    ).fetchall()
    return {str(row["name"]) for row in rows}


def scalar(connection, statement: str) -> int:
    return int(connection.execute(statement).fetchone()["total"] or 0)


def drop_column(connection, table: str, column: str) -> None:
    constraints = connection.execute(
        """SELECT DISTINCT CONSTRAINT_NAME AS name
        FROM information_schema.key_column_usage
        WHERE table_schema=DATABASE() AND table_name=%s AND column_name=%s
        AND referenced_table_name IS NOT NULL""",
        (table, column),
    ).fetchall()
    for row in constraints:
        connection.execute(f"ALTER TABLE `{table}` DROP FOREIGN KEY `{row['name']}`")
    indexes = connection.execute(
        """SELECT DISTINCT INDEX_NAME AS name FROM information_schema.statistics
        WHERE table_schema=DATABASE() AND table_name=%s AND column_name=%s
        AND INDEX_NAME <> 'PRIMARY'""",
        (table, column),
    ).fetchall()
    for row in indexes:
        connection.execute(f"ALTER TABLE `{table}` DROP INDEX `{row['name']}`")
    connection.execute(f"ALTER TABLE `{table}` DROP COLUMN `{column}`")


def drop_fk_for_column(connection, table: str, column: str) -> None:
    rows = connection.execute(
        """SELECT DISTINCT CONSTRAINT_NAME AS name
        FROM information_schema.key_column_usage
        WHERE table_schema=DATABASE() AND table_name=%s AND column_name=%s
        AND referenced_table_name IS NOT NULL""",
        (table, column),
    ).fetchall()
    for row in rows:
        connection.execute(f"ALTER TABLE `{table}` DROP FOREIGN KEY `{row['name']}`")


def validate(connection) -> dict[str, int]:
    session_columns = columns(connection, "monitoring_sessions")
    location_columns = columns(connection, "vehicle_locations")
    results = {
        "sessions_without_trip": scalar(connection, """SELECT COUNT(*) AS total
            FROM monitoring_sessions s LEFT JOIN trips t ON t.id=s.trip_id
            WHERE s.trip_id IS NULL OR t.id IS NULL"""),
        "session_driver_mismatches": 0,
        "session_vehicle_mismatches": 0,
        "location_vehicle_mismatches": 0,
    }
    if "driver_id" in session_columns:
        results["session_driver_mismatches"] = scalar(connection, """SELECT COUNT(*) AS total
            FROM monitoring_sessions s JOIN trips t ON t.id=s.trip_id
            WHERE s.driver_id IS NOT NULL AND s.driver_id<>t.driver_id""")
    if "vehicle_id" in session_columns:
        results["session_vehicle_mismatches"] = scalar(connection, """SELECT COUNT(*) AS total
            FROM monitoring_sessions s JOIN trips t ON t.id=s.trip_id
            WHERE s.vehicle_id IS NOT NULL AND s.vehicle_id<>t.vehicle_id""")
    if "vehicle_id" in location_columns:
        results["location_vehicle_mismatches"] = scalar(connection, """SELECT COUNT(*) AS total
            FROM vehicle_locations l JOIN trips t ON t.id=l.trip_id
            WHERE l.vehicle_id<>t.vehicle_id""")
    return results


def repair_legacy_sessions(connection) -> int:
    rows = connection.execute("""SELECT id, driver_id, vehicle_id, status,
        started_at, ended_at, created_at FROM monitoring_sessions
        WHERE trip_id IS NULL ORDER BY id""").fetchall()
    repaired = 0
    for session in rows:
        if session["driver_id"] is None:
            raise RuntimeError(f"Phiên #{session['id']} không có driver_id để phục hồi")
        driver = connection.execute(
            "SELECT id FROM drivers WHERE id=%s", (session["driver_id"],),
        ).fetchone()
        if driver is None:
            raise RuntimeError(f"Phiên #{session['id']} tham chiếu tài xế không tồn tại")

        vehicle_id = session["vehicle_id"]
        if vehicle_id is None:
            legacy_vehicle = connection.execute(
                "SELECT id FROM vehicles WHERE plate_number='LEGACY-UNASSIGNED'"
            ).fetchone()
            if legacy_vehicle is None:
                now = datetime.now(timezone.utc).replace(tzinfo=None)
                cursor = connection.execute("""INSERT INTO vehicles(
                    plate_number, vehicle_type, model, status, active, created_at, updated_at
                ) VALUES ('LEGACY-UNASSIGNED', 'legacy',
                    'Dữ liệu phiên trước khi chuẩn hóa', 'archived', 0, %s, %s)""", (now, now))
                vehicle_id = int(cursor.lastrowid)
            else:
                vehicle_id = int(legacy_vehicle["id"])

        code = f"LEGACY-SESSION-{session['id']}"
        trip = connection.execute(
            "SELECT id, driver_id, vehicle_id FROM trips WHERE trip_code=%s", (code,),
        ).fetchone()
        if trip is None:
            final_status = "completed" if session["status"] == "completed" else "cancelled"
            created = session["created_at"] or datetime.now(timezone.utc).replace(tzinfo=None)
            cursor = connection.execute("""INSERT INTO trips(
                trip_code, driver_id, vehicle_id, route_id, status, planned_start_at,
                started_at, ended_at, created_at, updated_at
            ) VALUES (%s, %s, %s, NULL, %s, %s, %s, %s, %s, %s)""", (
                code, session["driver_id"], vehicle_id, final_status,
                session["started_at"], session["started_at"], session["ended_at"],
                created, session["ended_at"] or created,
            ))
            trip_id = int(cursor.lastrowid)
        else:
            if (int(trip["driver_id"]) != int(session["driver_id"])
                    or int(trip["vehicle_id"]) != int(vehicle_id)):
                raise RuntimeError(f"Chuyến phục hồi {code} đã tồn tại nhưng không khớp")
            trip_id = int(trip["id"])
        connection.execute(
            "UPDATE monitoring_sessions SET trip_id=%s WHERE id=%s",
            (trip_id, session["id"]),
        )
        repaired += 1
    return repaired


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Chuẩn hóa monitoring_sessions và vehicle_locations theo trips."
    )
    parser.add_argument("--apply", action="store_true", help="Thực hiện ALTER TABLE sau khi kiểm tra")
    parser.add_argument(
        "--repair-legacy", action="store_true",
        help="Tạo chuyến lưu trữ cho phiên cũ có driver_id nhưng chưa có trip_id",
    )
    args = parser.parse_args()
    if args.repair_legacy and not args.apply:
        raise SystemExit("--repair-legacy chỉ được dùng cùng --apply sau khi đã sao lưu.")
    url = os.getenv("DMS_DATABASE_URL", "")
    if not url.startswith(("mysql://", "mysql+mysqlconnector://")):
        raise SystemExit("Hãy đặt DMS_DATABASE_URL trỏ tới MySQL.")

    database = Database(url)
    with database.connect() as connection:
        checks = validate(connection)
        print("Kết quả kiểm tra:")
        for name, count in checks.items():
            print(f"- {name}: {count}")
        if checks["sessions_without_trip"] and args.repair_legacy:
            repaired = repair_legacy_sessions(connection)
            print(f"Đã phục hồi quan hệ chuyến cho {repaired} phiên cũ.")
            checks = validate(connection)
            for name, count in checks.items():
                print(f"- sau phục hồi {name}: {count}")
        if any(checks.values()):
            raise SystemExit("Dừng migration: cần xử lý các bản ghi không nhất quán ở trên.")
        if not args.apply:
            print("PASS kiểm tra. Chạy lại với --apply để thực hiện migration.")
            return

        session_columns = columns(connection, "monitoring_sessions")
        location_columns = columns(connection, "vehicle_locations")
        for column in ("driver_id", "vehicle_id"):
            if column in session_columns:
                drop_column(connection, "monitoring_sessions", column)
        if "vehicle_id" in location_columns:
            drop_column(connection, "vehicle_locations", "vehicle_id")

        drop_fk_for_column(connection, "monitoring_sessions", "trip_id")
        connection.execute("ALTER TABLE monitoring_sessions MODIFY trip_id BIGINT NOT NULL")
        connection.execute("""ALTER TABLE monitoring_sessions
            ADD CONSTRAINT fk_sessions_trip FOREIGN KEY(trip_id)
            REFERENCES trips(id) ON DELETE RESTRICT""")
        print("PASS: schema đã chuẩn hóa; tài xế và xe được suy ra qua trips.")


if __name__ == "__main__":
    main()
