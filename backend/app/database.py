from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS drivers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id TEXT NOT NULL UNIQUE, display_name TEXT NOT NULL, phone TEXT,
    active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS edge_devices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_uid TEXT NOT NULL UNIQUE, label TEXT NOT NULL,
    api_key_hash TEXT NOT NULL, vehicle_id INTEGER,
    active INTEGER NOT NULL DEFAULT 1, status TEXT NOT NULL DEFAULT 'offline',
    last_seen_at TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    FOREIGN KEY(vehicle_id) REFERENCES vehicles(id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS app_users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL,
    display_name TEXT NOT NULL, role TEXT NOT NULL,
    driver_id INTEGER, active INTEGER NOT NULL DEFAULT 1,
    token_version INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    FOREIGN KEY(driver_id) REFERENCES drivers(id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS monitoring_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL, status TEXT NOT NULL, device TEXT NOT NULL,
    yolo_interval INTEGER NOT NULL DEFAULT 1, cnn_interval INTEGER NOT NULL DEFAULT 1,
    save_video INTEGER NOT NULL DEFAULT 1, started_at TEXT, ended_at TEXT,
    frame_count INTEGER NOT NULL DEFAULT 0, video_duration_seconds REAL NOT NULL DEFAULT 0,
    processing_seconds REAL NOT NULL DEFAULT 0, avg_fps REAL NOT NULL DEFAULT 0,
    avg_risk_score REAL NOT NULL DEFAULT 0, max_risk_score INTEGER NOT NULL DEFAULT 0,
    face_detection_rate REAL NOT NULL DEFAULT 0, output_video TEXT, error_message TEXT,
    trip_id INTEGER NOT NULL, edge_device_id INTEGER, created_at TEXT NOT NULL,
    FOREIGN KEY(trip_id) REFERENCES trips(id) ON DELETE RESTRICT,
    FOREIGN KEY(edge_device_id) REFERENCES edge_devices(id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS detection_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL,
    occurred_at TEXT NOT NULL, frame_index INTEGER NOT NULL,
    timestamp_seconds REAL NOT NULL, event_type TEXT NOT NULL,
    confidence REAL NOT NULL, risk_score INTEGER NOT NULL, snapshot_path TEXT, clip_path TEXT,
    client_event_id TEXT UNIQUE, event_payload TEXT,
    review_status TEXT NOT NULL DEFAULT 'new', review_note TEXT,
    reviewed_at TEXT, reviewed_by INTEGER,
    FOREIGN KEY(session_id) REFERENCES monitoring_sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_events_session_time
ON detection_events(session_id, timestamp_seconds);
CREATE TABLE IF NOT EXISTS frame_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL,
    frame_index INTEGER NOT NULL, timestamp_seconds REAL NOT NULL,
    ear_left REAL, ear_right REAL, ear REAL, mar REAL,
    geometry_eye_closed INTEGER, yawning INTEGER, blink_count INTEGER NOT NULL DEFAULT 0,
    yawn_count INTEGER NOT NULL DEFAULT 0, cnn_awake REAL, cnn_drowsy REAL,
    lstm_awake REAL, lstm_drowsy REAL, perclos REAL, phone INTEGER NOT NULL DEFAULT 0,
    cigarette INTEGER NOT NULL DEFAULT 0, seatbelt INTEGER NOT NULL DEFAULT 0,
    yolo_eye_closed INTEGER NOT NULL DEFAULT 0, yolo_eye_open INTEGER NOT NULL DEFAULT 0,
    risk_score INTEGER NOT NULL, severity TEXT NOT NULL, warnings TEXT NOT NULL,
    processing_fps REAL NOT NULL, pitch REAL, yaw REAL, roll REAL, gaze_x REAL,
    gaze_y REAL, eye_closure_seconds REAL NOT NULL DEFAULT 0,
    max_eye_closure_seconds REAL NOT NULL DEFAULT 0, microsleep INTEGER NOT NULL DEFAULT 0,
    head_distracted INTEGER NOT NULL DEFAULT 0, gaze_distracted INTEGER NOT NULL DEFAULT 0,
    distraction_score REAL NOT NULL DEFAULT 0,
    FOREIGN KEY(session_id) REFERENCES monitoring_sessions(id) ON DELETE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_frame_metrics_session_frame
ON frame_metrics(session_id, frame_index);
"""

MYSQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS drivers (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    external_id VARCHAR(100) NOT NULL UNIQUE, display_name VARCHAR(255) NOT NULL, phone VARCHAR(50),
    active TINYINT(1) NOT NULL DEFAULT 1, created_at DATETIME(6) NOT NULL, updated_at DATETIME(6) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS edge_devices (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    device_uid VARCHAR(255) NOT NULL UNIQUE, label VARCHAR(255) NOT NULL,
    api_key_hash VARCHAR(64) NOT NULL, vehicle_id BIGINT,
    active TINYINT(1) NOT NULL DEFAULT 1, status VARCHAR(20) NOT NULL DEFAULT 'offline',
    last_seen_at DATETIME(6), created_at DATETIME(6) NOT NULL, updated_at DATETIME(6) NOT NULL,
    INDEX idx_edge_devices_vehicle_id(vehicle_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS app_users (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(100) NOT NULL UNIQUE, password_hash VARCHAR(255) NOT NULL,
    display_name VARCHAR(255) NOT NULL, role VARCHAR(30) NOT NULL,
    driver_id BIGINT, active TINYINT(1) NOT NULL DEFAULT 1,
    token_version INTEGER NOT NULL DEFAULT 0,
    created_at DATETIME(6) NOT NULL, updated_at DATETIME(6) NOT NULL,
    CONSTRAINT fk_users_driver FOREIGN KEY(driver_id) REFERENCES drivers(id) ON DELETE SET NULL,
    INDEX idx_users_driver_id(driver_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS monitoring_sessions (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    source VARCHAR(500) NOT NULL, status VARCHAR(30) NOT NULL, device VARCHAR(255) NOT NULL,
    yolo_interval INTEGER NOT NULL DEFAULT 1, cnn_interval INTEGER NOT NULL DEFAULT 1,
    save_video TINYINT(1) NOT NULL DEFAULT 1, started_at DATETIME(6), ended_at DATETIME(6),
    frame_count INTEGER NOT NULL DEFAULT 0, video_duration_seconds DOUBLE NOT NULL DEFAULT 0,
    processing_seconds DOUBLE NOT NULL DEFAULT 0, avg_fps DOUBLE NOT NULL DEFAULT 0,
    avg_risk_score DOUBLE NOT NULL DEFAULT 0, max_risk_score INTEGER NOT NULL DEFAULT 0,
    face_detection_rate DOUBLE NOT NULL DEFAULT 0, output_video TEXT, error_message TEXT,
    trip_id BIGINT NOT NULL, edge_device_id BIGINT, created_at DATETIME(6) NOT NULL,
    INDEX idx_sessions_trip_id(trip_id),
    INDEX idx_sessions_edge_device_id(edge_device_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS detection_events (
    id BIGINT PRIMARY KEY AUTO_INCREMENT, session_id BIGINT NOT NULL,
    occurred_at DATETIME(6) NOT NULL, frame_index INTEGER NOT NULL,
    timestamp_seconds DOUBLE NOT NULL, event_type VARCHAR(100) NOT NULL,
    confidence DOUBLE NOT NULL, risk_score INTEGER NOT NULL, snapshot_path TEXT, clip_path TEXT,
    client_event_id VARCHAR(255) UNIQUE, event_payload JSON,
    review_status VARCHAR(30) NOT NULL DEFAULT 'new', review_note TEXT,
    reviewed_at DATETIME(6), reviewed_by BIGINT,
    severity VARCHAR(30), latitude DOUBLE, longitude DOUBLE,
    gps_source VARCHAR(80), gps_recorded_at DATETIME(6),
    CONSTRAINT fk_events_session FOREIGN KEY(session_id) REFERENCES monitoring_sessions(id) ON DELETE CASCADE,
    INDEX idx_events_session_time(session_id, timestamp_seconds)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS frame_metrics (
    id BIGINT PRIMARY KEY AUTO_INCREMENT, session_id BIGINT NOT NULL,
    frame_index INTEGER NOT NULL, timestamp_seconds DOUBLE NOT NULL,
    ear_left DOUBLE, ear_right DOUBLE, ear DOUBLE, mar DOUBLE,
    geometry_eye_closed TINYINT(1), yawning TINYINT(1), blink_count INTEGER NOT NULL DEFAULT 0,
    yawn_count INTEGER NOT NULL DEFAULT 0, cnn_awake DOUBLE, cnn_drowsy DOUBLE,
    lstm_awake DOUBLE, lstm_drowsy DOUBLE, perclos DOUBLE, phone TINYINT(1) NOT NULL DEFAULT 0,
    cigarette TINYINT(1) NOT NULL DEFAULT 0, seatbelt TINYINT(1) NOT NULL DEFAULT 0,
    yolo_eye_closed TINYINT(1) NOT NULL DEFAULT 0, yolo_eye_open TINYINT(1) NOT NULL DEFAULT 0,
    risk_score INTEGER NOT NULL, severity VARCHAR(30) NOT NULL, warnings VARCHAR(500) NOT NULL,
    processing_fps DOUBLE NOT NULL, pitch DOUBLE, yaw DOUBLE, roll DOUBLE, gaze_x DOUBLE,
    gaze_y DOUBLE, eye_closure_seconds DOUBLE NOT NULL DEFAULT 0,
    max_eye_closure_seconds DOUBLE NOT NULL DEFAULT 0, microsleep TINYINT(1) NOT NULL DEFAULT 0,
    head_distracted TINYINT(1) NOT NULL DEFAULT 0, gaze_distracted TINYINT(1) NOT NULL DEFAULT 0,
    distraction_score DOUBLE NOT NULL DEFAULT 0,
    CONSTRAINT fk_metrics_session FOREIGN KEY(session_id) REFERENCES monitoring_sessions(id) ON DELETE CASCADE,
    UNIQUE KEY uq_metrics_session_frame(session_id, frame_index)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

FLEET_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS vehicles (
    id INTEGER PRIMARY KEY AUTOINCREMENT, plate_number TEXT NOT NULL UNIQUE,
    vehicle_type TEXT NOT NULL, model TEXT, status TEXT NOT NULL DEFAULT 'available',
    active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS routes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, route_code TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL, start_location TEXT NOT NULL, end_location TEXT NOT NULL,
    distance_km REAL, active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trips (
    id INTEGER PRIMARY KEY AUTOINCREMENT, trip_code TEXT NOT NULL UNIQUE,
    driver_id INTEGER NOT NULL, vehicle_id INTEGER NOT NULL, route_id INTEGER,
    status TEXT NOT NULL DEFAULT 'planned', planned_start_at TEXT,
    started_at TEXT, ended_at TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    FOREIGN KEY(driver_id) REFERENCES drivers(id) ON DELETE RESTRICT,
    FOREIGN KEY(vehicle_id) REFERENCES vehicles(id) ON DELETE RESTRICT,
    FOREIGN KEY(route_id) REFERENCES routes(id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS vehicle_locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, trip_id INTEGER NOT NULL,
    latitude REAL NOT NULL, longitude REAL NOT NULL,
    speed_kph REAL, heading REAL, recorded_at TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'edge-simulator', created_at TEXT NOT NULL,
    FOREIGN KEY(trip_id) REFERENCES trips(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_trips_driver_status ON trips(driver_id, status);
CREATE INDEX IF NOT EXISTS idx_locations_trip_time ON vehicle_locations(trip_id, recorded_at);
"""

FLEET_MYSQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS vehicles (
    id BIGINT PRIMARY KEY AUTO_INCREMENT, plate_number VARCHAR(30) NOT NULL UNIQUE,
    vehicle_type VARCHAR(80) NOT NULL, model VARCHAR(120),
    status VARCHAR(30) NOT NULL DEFAULT 'available', active TINYINT(1) NOT NULL DEFAULT 1,
    created_at DATETIME(6) NOT NULL, updated_at DATETIME(6) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS routes (
    id BIGINT PRIMARY KEY AUTO_INCREMENT, route_code VARCHAR(80) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL, start_location VARCHAR(255) NOT NULL,
    end_location VARCHAR(255) NOT NULL, distance_km DOUBLE,
    active TINYINT(1) NOT NULL DEFAULT 1, created_at DATETIME(6) NOT NULL,
    updated_at DATETIME(6) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS trips (
    id BIGINT PRIMARY KEY AUTO_INCREMENT, trip_code VARCHAR(100) NOT NULL UNIQUE,
    driver_id BIGINT NOT NULL, vehicle_id BIGINT NOT NULL, route_id BIGINT,
    status VARCHAR(30) NOT NULL DEFAULT 'planned', planned_start_at DATETIME(6),
    started_at DATETIME(6), ended_at DATETIME(6), created_at DATETIME(6) NOT NULL,
    updated_at DATETIME(6) NOT NULL,
    CONSTRAINT fk_trips_driver FOREIGN KEY(driver_id) REFERENCES drivers(id) ON DELETE RESTRICT,
    CONSTRAINT fk_trips_vehicle FOREIGN KEY(vehicle_id) REFERENCES vehicles(id) ON DELETE RESTRICT,
    CONSTRAINT fk_trips_route FOREIGN KEY(route_id) REFERENCES routes(id) ON DELETE SET NULL,
    INDEX idx_trips_driver_status(driver_id, status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS vehicle_locations (
    id BIGINT PRIMARY KEY AUTO_INCREMENT, trip_id BIGINT NOT NULL,
    latitude DOUBLE NOT NULL, longitude DOUBLE NOT NULL, speed_kph DOUBLE, heading DOUBLE,
    recorded_at DATETIME(6) NOT NULL, source VARCHAR(80) NOT NULL DEFAULT 'edge-simulator',
    created_at DATETIME(6) NOT NULL,
    CONSTRAINT fk_locations_trip FOREIGN KEY(trip_id) REFERENCES trips(id) ON DELETE CASCADE,
    INDEX idx_locations_trip_time(trip_id, recorded_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

SQLITE_SESSION_COLUMNS = {
    "yolo_interval": "INTEGER NOT NULL DEFAULT 1",
    "cnn_interval": "INTEGER NOT NULL DEFAULT 1",
    "save_video": "INTEGER NOT NULL DEFAULT 1",
    "trip_id": "INTEGER",
    "edge_device_id": "INTEGER",
}
SQLITE_EVENT_COLUMNS = {
    "snapshot_path": "TEXT", "clip_path": "TEXT", "client_event_id": "TEXT", "event_payload": "TEXT",
    "review_status": "TEXT NOT NULL DEFAULT 'new'", "review_note": "TEXT",
    "reviewed_at": "TEXT", "reviewed_by": "INTEGER",
    "severity": "TEXT", "latitude": "REAL", "longitude": "REAL",
    "gps_source": "TEXT", "gps_recorded_at": "TEXT",
}
SQLITE_USER_COLUMNS = {"token_version": "INTEGER NOT NULL DEFAULT 0"}
SQLITE_METRIC_COLUMNS = {
    "pitch": "REAL", "yaw": "REAL", "roll": "REAL", "gaze_x": "REAL", "gaze_y": "REAL",
    "eye_closure_seconds": "REAL NOT NULL DEFAULT 0",
    "max_eye_closure_seconds": "REAL NOT NULL DEFAULT 0",
    "microsleep": "INTEGER NOT NULL DEFAULT 0",
    "head_distracted": "INTEGER NOT NULL DEFAULT 0",
    "gaze_distracted": "INTEGER NOT NULL DEFAULT 0",
    "distraction_score": "REAL NOT NULL DEFAULT 0",
}
MYSQL_SESSION_COLUMNS = {
    "yolo_interval": "INTEGER NOT NULL DEFAULT 1",
    "cnn_interval": "INTEGER NOT NULL DEFAULT 1",
    "save_video": "TINYINT(1) NOT NULL DEFAULT 1",
    "trip_id": "BIGINT",
    "edge_device_id": "BIGINT",
}
MYSQL_EVENT_COLUMNS = {
    "snapshot_path": "TEXT", "clip_path": "TEXT",
    "client_event_id": "VARCHAR(255)", "event_payload": "JSON",
    "review_status": "VARCHAR(30) NOT NULL DEFAULT 'new'", "review_note": "TEXT",
    "reviewed_at": "DATETIME(6)", "reviewed_by": "BIGINT",
    "severity": "VARCHAR(30)", "latitude": "DOUBLE", "longitude": "DOUBLE",
    "gps_source": "VARCHAR(80)", "gps_recorded_at": "DATETIME(6)",
}
MYSQL_USER_COLUMNS = {"token_version": "INTEGER NOT NULL DEFAULT 0"}
MYSQL_METRIC_COLUMNS = {
    "pitch": "DOUBLE", "yaw": "DOUBLE", "roll": "DOUBLE",
    "gaze_x": "DOUBLE", "gaze_y": "DOUBLE",
    "eye_closure_seconds": "DOUBLE NOT NULL DEFAULT 0",
    "max_eye_closure_seconds": "DOUBLE NOT NULL DEFAULT 0",
    "microsleep": "TINYINT(1) NOT NULL DEFAULT 0",
    "head_distracted": "TINYINT(1) NOT NULL DEFAULT 0",
    "gaze_distracted": "TINYINT(1) NOT NULL DEFAULT 0",
    "distraction_score": "DOUBLE NOT NULL DEFAULT 0",
}
FRAME_DEFAULTS = {
    "blink_count": 0, "yawn_count": 0, "phone": 0, "cigarette": 0,
    "seatbelt": 0, "yolo_eye_closed": 0, "yolo_eye_open": 0,
    "eye_closure_seconds": 0, "max_eye_closure_seconds": 0, "microsleep": 0,
    "head_distracted": 0, "gaze_distracted": 0, "distraction_score": 0,
}
SESSION_FIELDS = (
    "id", "source", "status", "device", "yolo_interval", "cnn_interval",
    "save_video", "started_at", "ended_at", "frame_count",
    "video_duration_seconds", "processing_seconds", "avg_fps",
    "avg_risk_score", "max_risk_score", "face_detection_rate",
    "output_video", "error_message", "trip_id", "edge_device_id", "created_at",
)


def _session_projection(alias: str = "monitoring_sessions") -> str:
    return ", ".join(f"{alias}.{name}" for name in SESSION_FIELDS)


TEST_INDEX_SCHEMA = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_client_event_id
ON detection_events(client_event_id);
CREATE INDEX IF NOT EXISTS idx_edge_devices_vehicle_id
ON edge_devices(vehicle_id);
CREATE INDEX IF NOT EXISTS idx_users_driver_id
ON app_users(driver_id);
CREATE INDEX IF NOT EXISTS idx_sessions_trip_id
ON monitoring_sessions(trip_id);
"""


def _mysql_parameter(value: Any) -> Any:
    """Convert an ISO-8601 value to the UTC-naive value MySQL DATETIME stores."""
    if not isinstance(value, str) or len(value) < 19 or value[4:5] != "-":
        return value
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _api_value(value: Any) -> Any:
    """Expose MySQL DATETIME values through the same UTC ISO format as SQLite."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)
        return value.isoformat()
    return value


class _MySQLCursorAdapter:
    def __init__(self, cursor: Any):
        self._cursor = cursor

    def execute(self, statement: str, values: tuple | list | None = None):
        params = tuple(_mysql_parameter(value) for value in (values or ()))
        self._cursor.execute(statement, params)
        return self

    def executemany(self, statement: str, values: list[tuple]):
        params = [tuple(_mysql_parameter(value) for value in row) for row in values]
        self._cursor.executemany(statement, params)
        return self

    @staticmethod
    def _row(row: dict | None) -> dict | None:
        return {key: _api_value(value) for key, value in row.items()} if row else None

    def fetchone(self):
        return self._row(self._cursor.fetchone())

    def fetchall(self):
        return [self._row(row) for row in self._cursor.fetchall()]

    def __enter__(self):
        self._cursor.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._cursor.__exit__(exc_type, exc_value, traceback)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


class _MySQLConnectionAdapter:
    """Small adapter exposing the execute API already used by this module."""

    def __init__(self, connection: Any):
        self._connection = connection

    def execute(self, statement: str, values: tuple | list | None = None):
        return self.cursor().execute(statement, values)

    def executemany(self, statement: str, values: list[tuple]):
        return self.cursor().executemany(statement, values)

    def cursor(self):
        return _MySQLCursorAdapter(self._connection.cursor(dictionary=True))

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    def close(self) -> None:
        self._connection.close()


class Database:
    """Persistence layer for the MySQL application and isolated SQLite tests.

    Runtime configuration must use MySQL. A local path is accepted exclusively
    so unit tests can run without writing into the real MySQL database.
    """

    def __init__(self, database: str | Path):
        raw = str(database)
        self.is_mysql = raw.startswith(("mysql://", "mysql+mysqlconnector://"))
        if "://" in raw and not self.is_mysql and not raw.startswith("sqlite:///"):
            raise ValueError("Cơ sở dữ liệu runtime chỉ hỗ trợ MySQL")
        self.is_server_database = self.is_mysql
        self.url = raw
        self.path = None if self.is_server_database else self._sqlite_path(raw)

    @staticmethod
    def _sqlite_path(value: str) -> Path:
        if value.startswith("sqlite:///"):
            return Path(value.removeprefix("sqlite:///"))
        return Path(value)

    def _sql(self, statement: str) -> str:
        return statement.replace("?", "%s") if self.is_server_database else statement

    @contextmanager
    def connect(self) -> Iterator[Any]:
        if self.is_mysql:
            try:
                from mysql.connector import connect as mysql_connect
            except ImportError as error:  # pragma: no cover
                raise RuntimeError(
                    "Thiếu mysql-connector-python. Chạy: python -m pip install -r backend/requirements.txt"
                ) from error
            from urllib.parse import unquote, urlparse

            parsed = urlparse(self.url.replace("mysql+mysqlconnector://", "mysql://", 1))
            connection = _MySQLConnectionAdapter(mysql_connect(
                host=parsed.hostname or "127.0.0.1",
                port=parsed.port or 3306,
                user=unquote(parsed.username or ""),
                password=unquote(parsed.password or ""),
                database=parsed.path.lstrip("/"),
                charset="utf8mb4",
                connection_timeout=10,
            ))
        else:
            assert self.path is not None
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=30)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            # SQLite exists only for isolated unit tests.
            try:
                connection.execute("PRAGMA journal_mode = WAL")
            except sqlite3.OperationalError:
                pass
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _execute_schema(self, connection: Any, schema: str) -> None:
        for statement in schema.split(";"):
            statement = statement.strip()
            if statement:
                connection.execute(statement)

    def _existing_columns(self, connection: Any, table_name: str) -> set[str]:
        if self.is_mysql:
            rows = connection.execute(
                """SELECT COLUMN_NAME AS column_name FROM information_schema.columns
                WHERE table_schema = DATABASE() AND table_name = %s""",
                (table_name,),
            ).fetchall()
            return {row.get("column_name", row.get("COLUMN_NAME")) for row in rows}
        return {row["name"] for row in connection.execute(f"PRAGMA table_info({table_name})")}

    def _add_missing_columns(self, connection: Any, table_name: str, columns: dict[str, str]) -> None:
        existing = self._existing_columns(connection, table_name)
        for name, definition in columns.items():
            if name not in existing:
                connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {name} {definition}")

    def _ensure_fresh_mysql_relationships(self, connection: Any) -> None:
        """Finish FKs whose referenced fleet tables are created later in initialize()."""
        rows = connection.execute(
            """SELECT TABLE_NAME AS table_name, COLUMN_NAME AS column_name
            FROM information_schema.key_column_usage
            WHERE table_schema = DATABASE() AND referenced_table_name IS NOT NULL"""
        ).fetchall()
        foreign_keys = {(row["table_name"], row["column_name"]) for row in rows}
        if ("monitoring_sessions", "trip_id") not in foreign_keys:
            invalid = connection.execute(
                """SELECT COUNT(*) AS total FROM monitoring_sessions s
                LEFT JOIN trips t ON t.id=s.trip_id
                WHERE s.trip_id IS NULL OR t.id IS NULL"""
            ).fetchone()
            if int(invalid["total"] or 0) == 0:
                connection.execute(
                    "ALTER TABLE monitoring_sessions MODIFY trip_id BIGINT NOT NULL"
                )
                connection.execute(
                    """ALTER TABLE monitoring_sessions ADD CONSTRAINT fk_sessions_trip
                    FOREIGN KEY(trip_id) REFERENCES trips(id) ON DELETE RESTRICT"""
                )
        if ("monitoring_sessions", "edge_device_id") not in foreign_keys:
            connection.execute(
                """ALTER TABLE monitoring_sessions ADD CONSTRAINT fk_sessions_edge_device
                FOREIGN KEY(edge_device_id) REFERENCES edge_devices(id) ON DELETE SET NULL"""
            )

    def initialize(self) -> None:
        with self.connect() as connection:
            schema = MYSQL_SCHEMA if self.is_mysql else SQLITE_SCHEMA
            self._execute_schema(connection, schema)
            fleet_schema = (
                FLEET_MYSQL_SCHEMA if self.is_mysql else FLEET_SQLITE_SCHEMA
            )
            self._execute_schema(connection, fleet_schema)
            if self.is_mysql:
                self._ensure_fresh_mysql_relationships(connection)
            session_columns = MYSQL_SESSION_COLUMNS if self.is_mysql else SQLITE_SESSION_COLUMNS
            event_columns = MYSQL_EVENT_COLUMNS if self.is_mysql else SQLITE_EVENT_COLUMNS
            metric_columns = MYSQL_METRIC_COLUMNS if self.is_mysql else SQLITE_METRIC_COLUMNS
            user_columns = MYSQL_USER_COLUMNS if self.is_mysql else SQLITE_USER_COLUMNS
            self._add_missing_columns(connection, "monitoring_sessions", session_columns)
            self._add_missing_columns(connection, "detection_events", event_columns)
            self._add_missing_columns(connection, "frame_metrics", metric_columns)
            self._add_missing_columns(connection, "app_users", user_columns)
            if not self.is_mysql:  # MySQL indexes are declared in MYSQL_SCHEMA.
                self._execute_schema(connection, TEST_INDEX_SCHEMA)

    def _update_returning(
        self, connection: Any, statement: str, values: tuple[Any, ...],
        table_name: str, record_id: int,
    ) -> Any:
        """Execute UPDATE RETURNING on test SQLite and emulate it on MySQL."""
        if self.is_mysql:
            without_returning = statement.rsplit("RETURNING", 1)[0].rstrip()
            connection.execute(self._sql(without_returning), values)
            # MySQL reports rowcount=0 when the matched row already contains the
            # requested values. Selecting by id distinguishes that case from a
            # genuinely missing record.
            return connection.execute(
                self._sql(f"SELECT * FROM {table_name} WHERE id = ?"), (record_id,)
            ).fetchone()
        return connection.execute(self._sql(statement), values).fetchone()

    def _insert_record(
        self, connection: Any, statement: str, values: tuple[Any, ...], table_name: str,
    ) -> dict:
        cursor = connection.execute(self._sql(statement), values)
        row = connection.execute(
            self._sql(f"SELECT * FROM {table_name} WHERE id = ?"),
            (cursor.lastrowid,),
        ).fetchone()
        return dict(row)

    def create_vehicle(
        self, plate_number: str, vehicle_type: str, model: str | None = None,
    ) -> dict:
        now = utc_now()
        statement = """INSERT INTO vehicles(
            plate_number, vehicle_type, model, status, active, created_at, updated_at
        ) VALUES (?, ?, ?, 'available', ?, ?, ?)"""
        with self.connect() as connection:
            return self._insert_record(
                connection, statement,
                (plate_number.upper(), vehicle_type, model, True, now, now), "vehicles",
            )

    def list_vehicles(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                self._sql("SELECT * FROM vehicles WHERE plate_number <> ? ORDER BY plate_number, id"),
                ("LEGACY-UNASSIGNED",),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_vehicle(self, vehicle_id: int) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                self._sql("SELECT * FROM vehicles WHERE id = ?"), (vehicle_id,)
            ).fetchone()
        return dict(row) if row else None

    def update_vehicle(
        self, vehicle_id: int, vehicle_type: str, model: str | None,
        status: str, active: bool, plate_number: str | None = None,
    ) -> dict | None:
        statement = """UPDATE vehicles SET plate_number = COALESCE(?, plate_number),
            vehicle_type = ?, model = ?, status = ?, active = ?, updated_at = ?
            WHERE id = ? RETURNING *"""
        with self.connect() as connection:
            row = self._update_returning(
                connection, statement,
                (plate_number, vehicle_type, model, status, active, utc_now(), vehicle_id),
                "vehicles", vehicle_id,
            )
        return dict(row) if row else None

    def vehicle_reference_counts(self, vehicle_id: int) -> dict[str, int]:
        tables = ["edge_devices", "trips"]
        with self.connect() as connection:
            return {
                table: int(connection.execute(
                    self._sql(f"SELECT COUNT(*) AS total FROM {table} WHERE vehicle_id = ?"),
                    (vehicle_id,),
                ).fetchone()["total"])
                for table in tables
            }

    def delete_vehicle(self, vehicle_id: int) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                self._sql("DELETE FROM vehicles WHERE id = ?"), (vehicle_id,)
            )
        return cursor.rowcount > 0

    def create_route(
        self, route_code: str, name: str, start_location: str,
        end_location: str, distance_km: float | None,
    ) -> dict:
        now = utc_now()
        statement = """INSERT INTO routes(
            route_code, name, start_location, end_location, distance_km,
            active, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""
        with self.connect() as connection:
            return self._insert_record(
                connection, statement,
                (route_code, name, start_location, end_location, distance_km, True, now, now),
                "routes",
            )

    def list_routes(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM routes ORDER BY route_code, id").fetchall()
        return [dict(row) for row in rows]

    def get_route(self, route_id: int) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                self._sql("SELECT * FROM routes WHERE id = ?"), (route_id,)
            ).fetchone()
        return dict(row) if row else None

    def update_route(
        self, route_id: int, route_code: str, name: str, start_location: str,
        end_location: str, distance_km: float | None, active: bool,
    ) -> dict | None:
        statement = """UPDATE routes SET route_code = ?, name = ?, start_location = ?,
        end_location = ?, distance_km = ?, active = ?, updated_at = ?
        WHERE id = ? RETURNING *"""
        with self.connect() as connection:
            row = self._update_returning(
                connection, statement,
                (route_code, name, start_location, end_location, distance_km, active, utc_now(), route_id),
                "routes", route_id,
            )
        return dict(row) if row else None

    def route_trip_count(self, route_id: int) -> int:
        with self.connect() as connection:
            row = connection.execute(
                self._sql("SELECT COUNT(*) AS total FROM trips WHERE route_id = ?"), (route_id,)
            ).fetchone()
        return int(row["total"])

    def delete_route(self, route_id: int) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                self._sql("DELETE FROM routes WHERE id = ?"), (route_id,)
            )
        return cursor.rowcount > 0

    def create_trip(
        self, trip_code: str, driver_id: int, vehicle_id: int,
        route_id: int | None, planned_start_at: str | None,
    ) -> dict:
        now = utc_now()
        statement = """INSERT INTO trips(
            trip_code, driver_id, vehicle_id, route_id, status,
            planned_start_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, 'planned', ?, ?, ?)"""
        with self.connect() as connection:
            return self._insert_record(
                connection, statement,
                (trip_code, driver_id, vehicle_id, route_id, planned_start_at, now, now),
                "trips",
            )

    def get_trip(self, trip_id: int) -> dict | None:
        statement = """SELECT t.*, d.display_name AS driver_name,
            v.plate_number, r.name AS route_name
        FROM trips t JOIN drivers d ON d.id = t.driver_id
        JOIN vehicles v ON v.id = t.vehicle_id
        LEFT JOIN routes r ON r.id = t.route_id WHERE t.id = ?"""
        with self.connect() as connection:
            row = connection.execute(self._sql(statement), (trip_id,)).fetchone()
        return dict(row) if row else None

    def list_trips(self, status: str | None = None) -> list[dict]:
        statement = """SELECT t.*, d.display_name AS driver_name,
            v.plate_number, r.name AS route_name
        FROM trips t JOIN drivers d ON d.id = t.driver_id
        JOIN vehicles v ON v.id = t.vehicle_id
        LEFT JOIN routes r ON r.id = t.route_id
        WHERE t.trip_code NOT LIKE ?"""
        values: tuple[Any, ...] = ("LEGACY-SESSION-%",)
        if status is not None:
            statement += " AND t.status = ?"
            values = ("LEGACY-SESSION-%", status)
        statement += " ORDER BY t.id DESC"
        with self.connect() as connection:
            rows = connection.execute(self._sql(statement), values).fetchall()
        return [dict(row) for row in rows]

    def find_running_trip_conflict(
        self, driver_id: int, vehicle_id: int, exclude_trip_id: int,
    ) -> dict | None:
        statement = """SELECT t.*, d.display_name AS driver_name, v.plate_number
        FROM trips t JOIN drivers d ON d.id = t.driver_id
        JOIN vehicles v ON v.id = t.vehicle_id
        WHERE t.status = 'running' AND t.id <> ?
          AND (t.driver_id = ? OR t.vehicle_id = ?)
        ORDER BY t.id DESC LIMIT 1"""
        with self.connect() as connection:
            row = connection.execute(
                self._sql(statement), (exclude_trip_id, driver_id, vehicle_id)
            ).fetchone()
        return dict(row) if row else None

    def get_active_edge_device_for_vehicle(self, vehicle_id: int) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                self._sql("""SELECT * FROM edge_devices
                WHERE vehicle_id = ? AND active = 1
                ORDER BY id DESC LIMIT 1"""),
                (vehicle_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_driver_trips(self, driver_id: int) -> list[dict]:
        statement = """SELECT t.*, d.display_name AS driver_name,
            v.plate_number, r.name AS route_name
        FROM trips t JOIN drivers d ON d.id = t.driver_id
        JOIN vehicles v ON v.id = t.vehicle_id
        LEFT JOIN routes r ON r.id = t.route_id
        WHERE t.driver_id = ? ORDER BY t.id DESC"""
        with self.connect() as connection:
            rows = connection.execute(self._sql(statement), (driver_id,)).fetchall()
        return [dict(row) for row in rows]

    def get_driver_trip(self, driver_id: int, trip_id: int) -> dict | None:
        trip = self.get_trip(trip_id)
        return trip if trip is not None and int(trip["driver_id"]) == driver_id else None

    def update_trip_status(self, trip_id: int, status: str) -> dict | None:
        now = utc_now()
        time_column = {
            "running": "started_at", "completed": "ended_at", "cancelled": "ended_at",
        }.get(status)
        if time_column:
            statement = f"""UPDATE trips SET status = ?, {time_column} = ?,
                updated_at = ? WHERE id = ? RETURNING *"""
            values = (status, now, now, trip_id)
        else:
            statement = "UPDATE trips SET status = ?, updated_at = ? WHERE id = ? RETURNING *"
            values = (status, now, trip_id)
        with self.connect() as connection:
            row = self._update_returning(
                connection, statement, values, "trips", trip_id
            )
        return dict(row) if row else None

    def update_planned_trip(
        self, trip_id: int, trip_code: str, driver_id: int, vehicle_id: int,
        route_id: int | None, planned_start_at: str | None,
    ) -> dict | None:
        statement = """UPDATE trips SET trip_code = ?, driver_id = ?, vehicle_id = ?,
        route_id = ?, planned_start_at = ?, updated_at = ?
        WHERE id = ? AND status = 'planned' RETURNING *"""
        with self.connect() as connection:
            row = self._update_returning(
                connection, statement,
                (trip_code, driver_id, vehicle_id, route_id, planned_start_at, utc_now(), trip_id),
                "trips", trip_id,
            )
        return dict(row) if row else None

    def trip_dependency_counts(self, trip_id: int) -> dict[str, int]:
        with self.connect() as connection:
            return {
                "locations": int(connection.execute(
                    self._sql("SELECT COUNT(*) AS total FROM vehicle_locations WHERE trip_id = ?"),
                    (trip_id,),
                ).fetchone()["total"]),
                "sessions": int(connection.execute(
                    self._sql("SELECT COUNT(*) AS total FROM monitoring_sessions WHERE trip_id = ?"),
                    (trip_id,),
                ).fetchone()["total"]),
            }

    def delete_trip(self, trip_id: int) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                self._sql("DELETE FROM trips WHERE id = ?"), (trip_id,)
            )
        return cursor.rowcount > 0

    def delete_trip_with_dependencies(self, trip_id: int) -> list[int]:
        """Delete a trip together with its GPS points and monitoring records."""
        with self.connect() as connection:
            session_rows = connection.execute(
                self._sql("SELECT id FROM monitoring_sessions WHERE trip_id = ?"),
                (trip_id,),
            ).fetchall()
            session_ids = [int(row["id"]) for row in session_rows]
            connection.execute(
                self._sql("DELETE FROM monitoring_sessions WHERE trip_id = ?"), (trip_id,)
            )
            connection.execute(
                self._sql("DELETE FROM vehicle_locations WHERE trip_id = ?"), (trip_id,)
            )
            connection.execute(self._sql("DELETE FROM trips WHERE id = ?"), (trip_id,))
        return session_ids

    def add_vehicle_location(
        self, trip_id: int, latitude: float, longitude: float,
        speed_kph: float | None, heading: float | None,
        recorded_at: str | None, source: str,
    ) -> dict:
        with self.connect() as connection:
            # A legacy database may still contain the redundant NOT NULL vehicle_id
            # column until scripts/22_normalize_relationships.py is run.
            if "vehicle_id" in self._existing_columns(connection, "vehicle_locations"):
                trip = connection.execute(
                    self._sql("SELECT vehicle_id FROM trips WHERE id = ?"), (trip_id,),
                ).fetchone()
                if trip is None:
                    raise ValueError("Không tìm thấy chuyến đi")
                statement = """INSERT INTO vehicle_locations(
                    trip_id, vehicle_id, latitude, longitude, speed_kph, heading,
                    recorded_at, source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"""
                values = (
                    trip_id, int(trip["vehicle_id"]), latitude, longitude,
                    speed_kph, heading, recorded_at or utc_now(), source, utc_now(),
                )
            else:
                statement = """INSERT INTO vehicle_locations(
                    trip_id, latitude, longitude, speed_kph, heading,
                    recorded_at, source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""
                values = (
                    trip_id, latitude, longitude, speed_kph, heading,
                    recorded_at or utc_now(), source, utc_now(),
                )
            return self._insert_record(
                connection, statement, values, "vehicle_locations",
            )

    def list_trip_locations(self, trip_id: int, limit: int = 1000) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                self._sql("""SELECT * FROM vehicle_locations WHERE trip_id = ?
                ORDER BY recorded_at DESC, id DESC LIMIT ?"""), (trip_id, limit)
            ).fetchall()
        return [dict(row) for row in rows]

    def create_session(
        self, source: str, device: str, yolo_interval: int, cnn_interval: int,
        save_video: bool, trip_id: int,
        edge_device_id: int | None = None,
    ) -> int:
        statement = """INSERT INTO monitoring_sessions(
            source, status, device, yolo_interval, cnn_interval, save_video,
            trip_id, edge_device_id, created_at
        ) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?)"""
        values = (
            source, device, yolo_interval, cnn_interval, int(save_video),
            trip_id, edge_device_id, utc_now(),
        )
        with self.connect() as connection:
            return int(connection.execute(self._sql(statement), values).lastrowid)

    def update_session(self, session_id: int, **values: Any) -> None:
        if not values:
            return
        columns = ", ".join(f"{key} = ?" for key in values)
        with self.connect() as connection:
            connection.execute(
                self._sql(f"UPDATE monitoring_sessions SET {columns} WHERE id = ?"),
                (*values.values(), session_id),
            )

    def add_event(self, session_id: int, event: dict) -> int:
        occurred_at = utc_now()
        with self.connect() as connection:
            client_event_id = event.get("client_event_id")
            if client_event_id:
                existing = connection.execute(
                    self._sql("SELECT id FROM detection_events WHERE client_event_id = ?"),
                    (client_event_id,),
                ).fetchone()
                if existing:
                    return int(existing["id"])
            session = connection.execute(self._sql(
                "SELECT trip_id FROM monitoring_sessions WHERE id = ?"
            ), (session_id,)).fetchone()
            location = None
            if session and session["trip_id"] is not None:
                candidates = connection.execute(self._sql(
                    "SELECT * FROM vehicle_locations WHERE trip_id = ? ORDER BY id DESC LIMIT 100"
                ), (session["trip_id"],)).fetchall()
                now = datetime.fromisoformat(occurred_at)
                eligible = []
                for row in candidates:
                    try:
                        stamp = datetime.fromisoformat(row["recorded_at"].replace("Z", "+00:00"))
                        if stamp.tzinfo is not None and 0 <= (now - stamp).total_seconds() <= 60:
                            eligible.append((stamp, row))
                    except (ValueError, TypeError):
                        continue
                if eligible:
                    location = max(eligible, key=lambda item: item[0])[1]
            risk = event["risk_score"]
            statement = """INSERT INTO detection_events(
                session_id, occurred_at, frame_index, timestamp_seconds,
                event_type, confidence, risk_score, snapshot_path, clip_path,
                client_event_id, event_payload, severity, latitude, longitude,
                gps_source, gps_recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
            values = (
                session_id, occurred_at, event["frame_index"], event["timestamp_seconds"],
                event["type"], event["confidence"], risk, event.get("snapshot_path"),
                event.get("clip_path"), client_event_id,
                json.dumps(event.get("event_payload"), ensure_ascii=False) if event.get("event_payload") is not None else None,
                "danger" if risk >= 60 else "warning" if risk >= 25 else "normal",
                location["latitude"] if location else None,
                location["longitude"] if location else None,
                location["source"] if location else None,
                location["recorded_at"] if location else None,
            )
            return int(connection.execute(self._sql(statement), values).lastrowid)

    def update_event_evidence(self, event_id: int, field: str, relative_path: str) -> dict | None:
        if field not in {"snapshot_path", "clip_path"}:
            raise ValueError("Trường bằng chứng không hợp lệ")
        with self.connect() as connection:
            connection.execute(
                self._sql(f"UPDATE detection_events SET {field} = ? WHERE id = ?"),
                (relative_path, event_id),
            )
        return self.get_event(event_id)

    def get_active_edge_session(self, edge_device_id: int) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                self._sql("SELECT * FROM monitoring_sessions WHERE edge_device_id = ? AND status IN ('queued','running') ORDER BY id DESC LIMIT 1"),
                (edge_device_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_active_session(self) -> dict | None:
        """Return the newest running session, including externally run Edge sessions."""
        with self.connect() as connection:
            row = connection.execute(self._sql(
                "SELECT id FROM monitoring_sessions WHERE status IN ('queued','running') ORDER BY id DESC LIMIT 1"
            )).fetchone()
        return self.get_session(int(row["id"])) if row else None

    def trip_events(self, trip_id: int) -> list[dict]:
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(self._sql(
                """SELECT e.* FROM detection_events e JOIN monitoring_sessions s
                ON s.id = e.session_id WHERE s.trip_id = ? ORDER BY e.id"""
            ), (trip_id,)).fetchall()]

    def safety_records(self, driver_id: int | None = None) -> tuple[list[dict], list[dict]]:
        with self.connect() as connection:
            clause = " WHERE t.driver_id = ?" if driver_id is not None else ""
            args = (driver_id,) if driver_id is not None else ()
            sessions = [dict(row) for row in connection.execute(self._sql(
                f"""SELECT {_session_projection('s')}, t.driver_id, t.vehicle_id,
                d.display_name AS driver_name, r.name AS route_name
                FROM monitoring_sessions s JOIN trips t ON t.id=s.trip_id
                JOIN drivers d ON d.id=t.driver_id
                LEFT JOIN routes r ON r.id=t.route_id""" + clause
            ), args).fetchall()]
            events = [dict(row) for row in connection.execute(self._sql(
                """SELECT e.*, t.driver_id, t.vehicle_id,
                d.display_name AS driver_name, r.name AS route_name
                FROM detection_events e JOIN monitoring_sessions s ON s.id=e.session_id
                JOIN trips t ON t.id=s.trip_id JOIN drivers d ON d.id=t.driver_id
                LEFT JOIN routes r ON r.id=t.route_id""" + clause
            ), args).fetchall()]
        return sessions, events

    def create_or_update_driver(self, external_id: str, display_name: str, phone: str | None) -> dict:
        now = utc_now()
        statement = """INSERT INTO drivers(external_id, display_name, phone, active, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(external_id) DO UPDATE SET
            display_name = excluded.display_name, phone = excluded.phone,
            active = excluded.active, updated_at = excluded.updated_at
        RETURNING *"""
        with self.connect() as connection:
            values = (external_id, display_name, phone, True, now, now)
            if self.is_mysql:
                connection.execute(self._sql("""INSERT INTO drivers(
                    external_id, display_name, phone, active, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON DUPLICATE KEY UPDATE display_name = VALUES(display_name),
                    phone = VALUES(phone), active = VALUES(active), updated_at = VALUES(updated_at)"""), values)
                row = connection.execute(
                    self._sql("SELECT * FROM drivers WHERE external_id = ?"), (external_id,)
                ).fetchone()
            else:
                row = connection.execute(self._sql(statement), values).fetchone()
        return dict(row)

    def list_drivers(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM drivers ORDER BY display_name, id").fetchall()
        return [dict(row) for row in rows]

    def get_driver(self, driver_id: int) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(self._sql("SELECT * FROM drivers WHERE id = ?"), (driver_id,)).fetchone()
        return dict(row) if row else None

    def get_driver_by_external_id(self, external_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                self._sql("SELECT * FROM drivers WHERE external_id = ?"),
                (external_id,),
            ).fetchone()
        return dict(row) if row else None

    def create_or_update_user(
        self,
        username: str,
        password_hash: str,
        display_name: str,
        role: str,
        driver_id: int | None,
    ) -> dict:
        if role not in {"driver", "supervisor"}:
            raise ValueError("Vai trò phải là driver hoặc supervisor")
        if role == "driver" and driver_id is None:
            raise ValueError("Tài khoản driver phải liên kết với một tài xế")
        now = utc_now()
        statement = """INSERT INTO app_users(
            username, password_hash, display_name, role, driver_id, active,
            token_version, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(username) DO UPDATE SET
            password_hash = excluded.password_hash,
            display_name = excluded.display_name,
            role = excluded.role,
            driver_id = excluded.driver_id,
            active = excluded.active,
            token_version = app_users.token_version + 1,
            updated_at = excluded.updated_at
        RETURNING *"""
        with self.connect() as connection:
            values = (username, password_hash, display_name, role, driver_id, True, 0, now, now)
            if self.is_mysql:
                connection.execute(self._sql("""INSERT INTO app_users(
                    username, password_hash, display_name, role, driver_id, active,
                    token_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON DUPLICATE KEY UPDATE password_hash = VALUES(password_hash),
                    display_name = VALUES(display_name), role = VALUES(role),
                    driver_id = VALUES(driver_id), active = VALUES(active),
                    token_version = token_version + 1, updated_at = VALUES(updated_at)"""), values)
                row = connection.execute(
                    self._sql("SELECT * FROM app_users WHERE username = ?"), (username,)
                ).fetchone()
            else:
                row = connection.execute(self._sql(statement), values).fetchone()
        return dict(row)

    def get_user_by_username(self, username: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                self._sql("SELECT * FROM app_users WHERE username = ?"),
                (username,),
            ).fetchone()
        return dict(row) if row else None

    def get_user(self, user_id: int) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                self._sql("SELECT * FROM app_users WHERE id = ?"),
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_user_by_driver_id(self, driver_id: int) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                self._sql("SELECT * FROM app_users WHERE driver_id = ? ORDER BY id LIMIT 1"),
                (driver_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_users(self) -> list[dict]:
        statement = """SELECT
            app_users.id, app_users.username, app_users.display_name,
            app_users.role, app_users.driver_id, app_users.active,
            app_users.created_at, app_users.updated_at,
            drivers.external_id AS driver_external_id,
            drivers.display_name AS driver_name
        FROM app_users
        LEFT JOIN drivers ON drivers.id = app_users.driver_id
        ORDER BY app_users.role DESC, app_users.display_name, app_users.id"""
        with self.connect() as connection:
            rows = connection.execute(statement).fetchall()
        return [dict(row) for row in rows]

    def set_user_active(self, user_id: int, active: bool) -> dict | None:
        statement = """UPDATE app_users
        SET active = ?, token_version = token_version + 1, updated_at = ?
        WHERE id = ?
        RETURNING *"""
        with self.connect() as connection:
            row = self._update_returning(
                connection, statement, (active, utc_now(), user_id), "app_users", user_id
            )
        return dict(row) if row else None

    def update_user_password(self, user_id: int, password_hash: str) -> dict | None:
        statement = """UPDATE app_users
        SET password_hash = ?, token_version = token_version + 1, updated_at = ?
        WHERE id = ?
        RETURNING *"""
        with self.connect() as connection:
            row = self._update_returning(
                connection, statement, (password_hash, utc_now(), user_id), "app_users", user_id
            )
        return dict(row) if row else None

    def update_driver_profile(
        self, driver_id: int, display_name: str, phone: str | None
    ) -> dict | None:
        statement = """UPDATE drivers
        SET display_name = ?, phone = ?, updated_at = ?
        WHERE id = ?
        RETURNING *"""
        with self.connect() as connection:
            row = self._update_returning(
                connection, statement, (display_name, phone, utc_now(), driver_id), "drivers", driver_id
            )
            if row is not None:
                connection.execute(
                    self._sql("UPDATE app_users SET display_name = ?, updated_at = ? WHERE driver_id = ?"),
                    (display_name, utc_now(), driver_id),
                )
        return dict(row) if row else None

    def create_edge_device(
        self, device_uid: str, label: str, api_key_hash: str,
        vehicle_id: int | None = None,
    ) -> dict:
        now = utc_now()
        statement = """INSERT INTO edge_devices(
            device_uid, label, api_key_hash, vehicle_id, active, status,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, 'offline', ?, ?)"""
        with self.connect() as connection:
            return self._insert_record(
                connection, statement,
                (device_uid, label, api_key_hash, vehicle_id, True, now, now),
                "edge_devices",
            )

    def list_edge_devices(self) -> list[dict]:
        statement = """SELECT edge_devices.*, vehicles.plate_number
        FROM edge_devices LEFT JOIN vehicles ON vehicles.id = edge_devices.vehicle_id
        ORDER BY edge_devices.updated_at DESC, edge_devices.id DESC"""
        with self.connect() as connection:
            rows = connection.execute(statement).fetchall()
        return [{key: value for key, value in dict(row).items() if key != "api_key_hash"} for row in rows]

    def get_edge_device(self, device_id: int) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                self._sql("SELECT * FROM edge_devices WHERE id = ?"), (device_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_edge_device_by_uid(self, device_uid: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                self._sql("SELECT * FROM edge_devices WHERE device_uid = ?"), (device_uid,)
            ).fetchone()
        return dict(row) if row else None

    def set_edge_device_active(self, device_id: int, active: bool) -> dict | None:
        statement = """UPDATE edge_devices
        SET active = ?, status = 'offline', updated_at = ?
        WHERE id = ?
        RETURNING *"""
        with self.connect() as connection:
            row = self._update_returning(
                connection, statement, (active, utc_now(), device_id),
                "edge_devices", device_id,
            )
        return {key: value for key, value in dict(row).items() if key != "api_key_hash"} if row else None

    def update_edge_device(
        self, device_id: int, label: str, vehicle_id: int | None,
    ) -> dict | None:
        statement = """UPDATE edge_devices SET label = ?, vehicle_id = ?,
        status = 'offline', updated_at = ? WHERE id = ? RETURNING *"""
        with self.connect() as connection:
            row = self._update_returning(
                connection, statement, (label, vehicle_id, utc_now(), device_id),
                "edge_devices", device_id,
            )
        return {key: value for key, value in dict(row).items() if key != "api_key_hash"} if row else None

    def delete_edge_device(self, device_id: int) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                self._sql("DELETE FROM edge_devices WHERE id = ?"), (device_id,)
            )
        return cursor.rowcount > 0

    def rotate_edge_device_key(self, device_id: int, api_key_hash: str) -> dict | None:
        statement = """UPDATE edge_devices SET api_key_hash = ?, status = 'offline',
        updated_at = ? WHERE id = ? RETURNING *"""
        with self.connect() as connection:
            row = self._update_returning(
                connection, statement, (api_key_hash, utc_now(), device_id),
                "edge_devices", device_id,
            )
        return {key: value for key, value in dict(row).items() if key != "api_key_hash"} if row else None

    def touch_edge_device(self, device_id: int) -> dict | None:
        now = utc_now()
        statement = """UPDATE edge_devices SET status = 'online', last_seen_at = ?,
        updated_at = ? WHERE id = ? AND active = 1 RETURNING *"""
        with self.connect() as connection:
            row = self._update_returning(
                connection, statement, (now, now, device_id), "edge_devices", device_id,
            )
        return dict(row) if row and bool(row["active"]) else None

    def mark_stale_edge_devices_offline(self, timeout_seconds: int = 60) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=timeout_seconds)).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                self._sql("""UPDATE edge_devices SET status = 'offline', updated_at = ?
                WHERE status = 'online' AND (last_seen_at IS NULL OR last_seen_at < ?)"""),
                (utc_now(), cutoff),
            )
        return cursor.rowcount

    def recover_interrupted_local_sessions(self) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                self._sql("""UPDATE monitoring_sessions
                SET status = 'interrupted', ended_at = ?,
                    error_message = 'Backend đã khởi động lại trước khi phiên hoàn tất'
                WHERE status IN ('queued', 'running')"""),
                (utc_now(),),
            )
        return cursor.rowcount

    def get_session(self, session_id: int) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                self._sql(f"""SELECT {_session_projection()},
                    trips.driver_id, trips.vehicle_id,
                    drivers.display_name AS driver_name,
                    drivers.external_id AS driver_external_id,
                    vehicles.plate_number,
                    trips.trip_code
                FROM monitoring_sessions
                JOIN trips ON trips.id = monitoring_sessions.trip_id
                JOIN drivers ON drivers.id = trips.driver_id
                JOIN vehicles ON vehicles.id = trips.vehicle_id
                WHERE monitoring_sessions.id = ?"""), (session_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_sessions(self, limit: int = 50, driver_id: int | None = None) -> list[dict]:
        statement = f"""SELECT {_session_projection()},
            trips.driver_id, trips.vehicle_id,
            drivers.display_name AS driver_name,
            drivers.external_id AS driver_external_id,
            vehicles.plate_number,
            trips.trip_code
        FROM monitoring_sessions
        JOIN trips ON trips.id = monitoring_sessions.trip_id
        JOIN drivers ON drivers.id = trips.driver_id
        JOIN vehicles ON vehicles.id = trips.vehicle_id"""
        values: tuple[Any, ...]
        if driver_id is not None:
            statement += " WHERE trips.driver_id = ?"
            values = (driver_id, limit)
        else:
            values = (limit,)
        statement += " ORDER BY monitoring_sessions.id DESC LIMIT ?"
        with self.connect() as connection:
            rows = connection.execute(
                self._sql(statement), values
            ).fetchall()
        return [dict(row) for row in rows]

    def list_driver_sessions(self, driver_id: int, limit: int = 20) -> list[dict]:
        """Return monitoring sessions belonging to one driver."""
        with self.connect() as connection:
            rows = connection.execute(
                self._sql(
                    f"SELECT {_session_projection('s')}, t.driver_id, t.vehicle_id "
                    "FROM monitoring_sessions s JOIN trips t ON t.id = s.trip_id "
                    "WHERE t.driver_id = ? "
                    "ORDER BY s.id DESC LIMIT ?"
                ),
                (driver_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_driver_recent_events(self, driver_id: int, limit: int = 30) -> list[dict]:
        statement = """SELECT detection_events.*, monitoring_sessions.started_at AS session_started_at,
            drivers.display_name AS driver_name, drivers.external_id AS driver_external_id,
            vehicles.plate_number
        FROM detection_events
        JOIN monitoring_sessions ON monitoring_sessions.id = detection_events.session_id
        JOIN trips ON trips.id = monitoring_sessions.trip_id
        JOIN drivers ON drivers.id = trips.driver_id
        JOIN vehicles ON vehicles.id = trips.vehicle_id
        WHERE trips.driver_id = ?
        ORDER BY detection_events.occurred_at DESC, detection_events.id DESC
        LIMIT ?"""
        with self.connect() as connection:
            rows = connection.execute(self._sql(statement), (driver_id, limit)).fetchall()
        return [dict(row) for row in rows]

    def driver_event_counts(self, driver_id: int) -> dict[str, int]:
        statement = """SELECT detection_events.event_type, COUNT(*) AS event_count
        FROM detection_events
        JOIN monitoring_sessions ON monitoring_sessions.id = detection_events.session_id
        JOIN trips ON trips.id = monitoring_sessions.trip_id
        WHERE trips.driver_id = ?
        GROUP BY detection_events.event_type"""
        with self.connect() as connection:
            rows = connection.execute(self._sql(statement), (driver_id,)).fetchall()
        return {str(row["event_type"]): int(row["event_count"]) for row in rows}

    def driver_review_counts(self, driver_id: int) -> dict[str, int]:
        statement = """SELECT COALESCE(detection_events.review_status, 'new') AS review_status,
            COUNT(*) AS event_count
        FROM detection_events
        JOIN monitoring_sessions ON monitoring_sessions.id = detection_events.session_id
        JOIN trips ON trips.id = monitoring_sessions.trip_id
        WHERE trips.driver_id = ?
        GROUP BY COALESCE(detection_events.review_status, 'new')"""
        with self.connect() as connection:
            rows = connection.execute(self._sql(statement), (driver_id,)).fetchall()
        counts = {"new": 0, "acknowledged": 0, "resolved": 0}
        for row in rows:
            counts[str(row["review_status"])] = int(row["event_count"])
        return counts

    def list_supervisor_alerts(
        self, limit: int = 100, review_status: str | None = None,
    ) -> list[dict]:
        statement = """SELECT detection_events.*,
            trips.driver_id,
            monitoring_sessions.source,
            monitoring_sessions.started_at AS session_started_at,
            drivers.display_name AS driver_name,
            drivers.external_id AS driver_external_id,
            app_users.display_name AS reviewed_by_name
        FROM detection_events
        JOIN monitoring_sessions ON monitoring_sessions.id = detection_events.session_id
        JOIN trips ON trips.id = monitoring_sessions.trip_id
        JOIN drivers ON drivers.id = trips.driver_id
        LEFT JOIN app_users ON app_users.id = detection_events.reviewed_by"""
        if review_status is not None:
            statement += " WHERE COALESCE(detection_events.review_status, 'new') = ?"
            values: tuple[Any, ...] = (review_status, limit)
        else:
            values = (limit,)
        statement += " ORDER BY detection_events.occurred_at DESC, detection_events.id DESC LIMIT ?"
        with self.connect() as connection:
            rows = connection.execute(self._sql(statement), values).fetchall()
        return [dict(row) for row in rows]

    def delete_session(self, session_id: int) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                self._sql("DELETE FROM monitoring_sessions WHERE id = ?"), (session_id,)
            )
        return cursor.rowcount > 0

    def list_events(self, session_id: int) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                self._sql("SELECT * FROM detection_events WHERE session_id = ? ORDER BY timestamp_seconds, id"),
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def latest_event_times(self, session_id: int) -> dict[str, float]:
        """Return the latest video timestamp per event type for cooldown recovery."""
        with self.connect() as connection:
            rows = connection.execute(
                self._sql(
                    "SELECT event_type, MAX(timestamp_seconds) AS latest_timestamp "
                    "FROM detection_events WHERE session_id = ? GROUP BY event_type"
                ),
                (session_id,),
            ).fetchall()
        return {
            str(row["event_type"]): float(row["latest_timestamp"] or 0.0)
            for row in rows
        }

    def get_event(self, event_id: int) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                self._sql("SELECT * FROM detection_events WHERE id = ?"), (event_id,)
            ).fetchone()
        return dict(row) if row else None

    def review_event(
        self, event_id: int, review_status: str, review_note: str | None,
        reviewed_by: int,
    ) -> dict | None:
        reviewed_at = None if review_status == "new" else utc_now()
        reviewer = None if review_status == "new" else reviewed_by
        statement = """UPDATE detection_events
        SET review_status = ?, review_note = ?, reviewed_at = ?, reviewed_by = ?
        WHERE id = ?
        RETURNING *"""
        with self.connect() as connection:
            row = self._update_returning(
                connection, statement,
                (review_status, review_note, reviewed_at, reviewer, event_id),
                "detection_events", event_id,
            )
        return dict(row) if row else None

    def add_frame_metrics(self, session_id: int, samples: list[dict]) -> None:
        if not samples:
            return
        columns = (
            "frame_index", "timestamp_seconds", "ear_left", "ear_right", "ear", "mar",
            "geometry_eye_closed", "yawning", "blink_count", "yawn_count",
            "cnn_awake", "cnn_drowsy", "lstm_awake", "lstm_drowsy", "perclos",
            "phone", "cigarette", "seatbelt", "yolo_eye_closed", "yolo_eye_open",
            "risk_score", "severity", "warnings", "processing_fps", "pitch", "yaw",
            "roll", "gaze_x", "gaze_y", "eye_closure_seconds", "max_eye_closure_seconds",
            "microsleep", "head_distracted", "gaze_distracted", "distraction_score",
        )
        placeholders = ", ".join("?" for _ in range(len(columns) + 1))
        statement = (
            f"INSERT OR REPLACE INTO frame_metrics(session_id, {', '.join(columns)}) "
            f"VALUES ({placeholders})"
        )
        if self.is_mysql:
            updated = ", ".join(f"{column} = VALUES({column})" for column in columns)
            statement = (
                f"INSERT INTO frame_metrics(session_id, {', '.join(columns)}) "
                f"VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {updated}"
            )
        values = [
            (session_id, *(sample.get(column, FRAME_DEFAULTS.get(column)) for column in columns))
            for sample in samples
        ]
        with self.connect() as connection:
            if self.is_server_database:
                with connection.cursor() as cursor:
                    cursor.executemany(self._sql(statement), values)
            else:
                connection.executemany(statement, values)

    def list_frame_metrics(self, session_id: int) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                self._sql("SELECT * FROM frame_metrics WHERE session_id = ? ORDER BY frame_index"),
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def frame_metrics_summary(self, session_id: int) -> dict:
        statement = """SELECT
            COUNT(*) AS sample_count, MIN(ear) AS ear_min, AVG(ear) AS ear_mean,
            MAX(ear) AS ear_max, MIN(mar) AS mar_min, AVG(mar) AS mar_mean,
            MAX(mar) AS mar_max, AVG(perclos) AS perclos_mean, MAX(perclos) AS perclos_max,
            MAX(blink_count) AS blink_count, MAX(yawn_count) AS yawn_count,
            AVG(cnn_drowsy) AS cnn_drowsy_mean, AVG(lstm_drowsy) AS lstm_drowsy_mean,
            AVG(processing_fps) AS processing_fps_mean, AVG(pitch) AS pitch_mean,
            MIN(pitch) AS pitch_min, MAX(pitch) AS pitch_max, AVG(yaw) AS yaw_mean,
            MIN(yaw) AS yaw_min, MAX(yaw) AS yaw_max, AVG(roll) AS roll_mean,
            AVG(gaze_x) AS gaze_x_mean, AVG(gaze_y) AS gaze_y_mean,
            MAX(max_eye_closure_seconds) AS max_eye_closure_seconds,
            SUM(microsleep) AS microsleep_samples,
            AVG(distraction_score) AS distraction_score_mean,
            MAX(distraction_score) AS distraction_score_max
            FROM frame_metrics WHERE session_id = ?"""
        with self.connect() as connection:
            row = connection.execute(self._sql(statement), (session_id,)).fetchone()
        return dict(row)

    def metrics_summary(self) -> dict:
        with self.connect() as connection:
            session = connection.execute("""SELECT
                COUNT(*) AS total_sessions,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_sessions,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_sessions,
                COALESCE(SUM(frame_count), 0) AS total_frames,
                COALESCE(AVG(CASE WHEN avg_fps > 0 THEN avg_fps END), 0) AS average_fps,
                COALESCE(MAX(max_risk_score), 0) AS maximum_risk_score
                FROM monitoring_sessions""").fetchone()
            event_rows = connection.execute(
                "SELECT event_type, COUNT(*) AS count FROM detection_events GROUP BY event_type ORDER BY count DESC"
            ).fetchall()
            total_events = connection.execute("SELECT COUNT(*) AS count FROM detection_events").fetchone()["count"]
        return {
            **dict(session), "total_events": int(total_events),
            "events_by_type": {row["event_type"]: row["count"] for row in event_rows},
        }
