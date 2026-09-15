from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.database import Database


def main() -> None:
    url = os.getenv("DMS_DATABASE_URL", "")
    if not url.startswith(("mysql://", "mysql+mysqlconnector://")):
        raise SystemExit("Hãy đặt DMS_DATABASE_URL trỏ tới MySQL trước khi migration.")
    database = Database(url)
    database.initialize()
    with database.connect() as connection:
        old_table = connection.execute(
            """SELECT COUNT(*) AS total FROM information_schema.tables
            WHERE table_schema = DATABASE() AND table_name = 'registered_devices'"""
        ).fetchone()
        if not old_table or int(old_table["total"]) == 0:
            print("PASS: edge_devices đã sẵn sàng; không có bảng legacy.")
            return
        old_rows = connection.execute("SELECT COUNT(*) AS total FROM registered_devices").fetchone()
        if int(old_rows["total"]) > 0:
            raise SystemExit(
                "DỪNG AN TOÀN: registered_devices còn dữ liệu. Hãy backup MySQL rồi xóa dữ liệu cũ "
                "trước khi chạy lại. Không tự chuyển mobile device thành Edge device."
            )
        connection.execute("DROP TABLE registered_devices")
    print("PASS: đã tạo edge_devices và xóa bảng registered_devices rỗng.")


if __name__ == "__main__":
    main()
