from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from backend.app.config import settings  # noqa: E402
from backend.app.database import Database  # noqa: E402
from backend.app.security import hash_password  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tạo hoặc đổi mật khẩu tài khoản DMS")
    parser.add_argument("--username", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--role", choices=("driver", "supervisor"), required=True)
    parser.add_argument("--driver-code", help="external_id, bắt buộc với role driver")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    database = Database(settings.database_url)
    database.initialize()
    driver_id = None
    if args.role == "driver":
        if not args.driver_code:
            raise SystemExit("--driver-code là bắt buộc với tài khoản driver")
        driver = database.get_driver_by_external_id(args.driver_code)
        if driver is None:
            driver = database.create_or_update_driver(
                args.driver_code,
                args.display_name.strip(),
                None,
            )
        driver_id = int(driver["id"])
    password = getpass.getpass("Mật khẩu mới (tối thiểu 8 ký tự): ")
    confirmation = getpass.getpass("Nhập lại mật khẩu: ")
    if len(password) < 8:
        raise SystemExit("Mật khẩu phải có ít nhất 8 ký tự")
    if password != confirmation:
        raise SystemExit("Hai mật khẩu không khớp")
    user = database.create_or_update_user(
        args.username.strip().lower(),
        hash_password(password),
        args.display_name.strip(),
        args.role,
        driver_id,
    )
    print(f"Đã lưu tài khoản {user['username']} ({user['role']})")


if __name__ == "__main__":
    main()
