from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Backup MySQL DMS chạy trong Docker Compose")
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "backups" / "mysql",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / f"dms-{datetime.now():%Y%m%d-%H%M%S}.sql"
    raw_url = os.getenv("DMS_DATABASE_URL", "")
    if not raw_url.startswith(("mysql://", "mysql+mysqlconnector://")):
        raise SystemExit("Hãy đặt DMS_DATABASE_URL trỏ tới MySQL trước khi backup.")
    parsed = urlparse(raw_url.replace("mysql+mysqlconnector://", "mysql://", 1))
    database_name = parsed.path.lstrip("/")
    bundled = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "MySQL" / "MySQL Server 8.0" / "bin" / "mysqldump.exe"
    executable = shutil.which("mysqldump") or (str(bundled) if bundled.is_file() else None)
    if executable is None:
        raise SystemExit("Không tìm thấy mysqldump của MySQL Server 8.0.")
    command = [
        executable, "--protocol=TCP", "--host", parsed.hostname or "127.0.0.1",
        "--port", str(parsed.port or 3306), "--user", unquote(parsed.username or ""),
        "--default-character-set=utf8mb4", "--single-transaction",
        "--routines", "--triggers", database_name,
    ]
    process_env = os.environ.copy()
    process_env["MYSQL_PWD"] = unquote(parsed.password or "")
    with destination.open("wb") as output:
        result = subprocess.run(
            command, stdout=output, stderr=subprocess.PIPE, check=False, env=process_env
        )
    if result.returncode != 0:
        destination.unlink(missing_ok=True)
        raise SystemExit(result.stderr.decode("utf-8", errors="replace"))
    print(f"MySQL backup: {destination.resolve()}")


if __name__ == "__main__":
    main()
