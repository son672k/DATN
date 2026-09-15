from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import settings
from backend.app.database import Database


def main() -> None:
    if not settings.database_url.startswith(("mysql://", "mysql+mysqlconnector://")):
        raise SystemExit(
            "DMS_DATABASE_URL chưa trỏ tới MySQL. Ví dụ:\n"
            "set DMS_DATABASE_URL=mysql://dms_app:change-me-before-deploying@127.0.0.1:3306/dms"
        )
    database = Database(settings.database_url)
    database.initialize()
    with database.connect() as connection:
        version = connection.execute("SELECT VERSION() AS version").fetchone()["version"]
        tables = connection.execute(
            """SELECT TABLE_NAME AS table_name FROM information_schema.tables
            WHERE table_schema = DATABASE() ORDER BY table_name"""
        ).fetchall()
    print(json.dumps({
        "status": "PASS",
        "database": "mysql",
        "version": version,
        "tables": [row.get("table_name", row.get("TABLE_NAME")) for row in tables],
        "url": settings.database_url.rsplit("@", 1)[-1],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
