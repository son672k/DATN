from __future__ import annotations

import argparse
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_ROOT = PROJECT_ROOT / "web_dashboard"
EDGE_LAUNCHER = PROJECT_ROOT / "scripts" / "21_edge_demo_launcher.py"
DEFAULT_BACKEND_PORT = 8000
DEFAULT_DASHBOARD_PORT = 3000

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def find_free_port(preferred_port: int, max_tries: int = 20) -> int:
    for offset in range(max_tries):
        candidate = preferred_port + offset
        available = True
        for family, host in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
            try:
                with socket.socket(family, socket.SOCK_STREAM) as sock:
                    sock.bind((host, candidate))
            except OSError:
                available = False
                break
        if available:
            return candidate
    raise RuntimeError(f"Không tìm được cổng trống bắt đầu từ {preferred_port}")


def wait_for_url(url: str, timeout_seconds: float = 30.0) -> bool:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return 200 <= response.status < 400
        except Exception as error:
            last_error = error
            time.sleep(0.5)
    if last_error is not None:
        print(f"Readiness probe thất bại cho {url}: {last_error}", file=sys.stderr)
    return False


def url_is_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return 200 <= response.status < 400
    except Exception:
        return False


def wait_for_backend(port: int, timeout_seconds: float = 30.0) -> bool:
    return wait_for_url(f"http://127.0.0.1:{port}/api/health", timeout_seconds)


def wait_for_process_url(process: subprocess.Popen, url: str, timeout_seconds: float) -> bool:
    """Require both the new process and its HTTP endpoint to be alive."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 400:
                    # Tránh nhận nhầm phản hồi của tiến trình cũ đúng lúc tiến
                    # trình vừa tạo đang thoát vì trùng cổng hoặc lỗi cấu hình.
                    time.sleep(0.5)
                    return process.poll() is None
        except Exception:
            pass
        time.sleep(0.5)
    return False


def terminate(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    if os.name == "nt":
        # npm.cmd launches vinext as a child. Terminating only npm leaves the
        # Node child holding the dashboard port, so always stop the exact tree.
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            pass
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description="Khởi động DMS backend và web dashboard")
    parser.add_argument(
        "--lan",
        action="store_true",
        help="Cho phép điện thoại cùng Wi-Fi truy cập FastAPI qua IPv4 LAN của máy tính",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Khởi động, xác minh API + dashboard rồi tự dừng",
    )
    parser.add_argument(
        "--no-edge-launcher",
        action="store_true",
        help="Không mở cửa sổ thiết lập Edge Simulator",
    )
    args = parser.parse_args()
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if npm is None:
        raise SystemExit("Không tìm thấy npm. Hãy cài Node.js LTS trước.")
    if not (DASHBOARD_ROOT / "node_modules").is_dir():
        raise SystemExit("Thiếu web_dashboard/node_modules. Chạy: cd web_dashboard; npm install")

    preferred_api = f"http://127.0.0.1:{DEFAULT_BACKEND_PORT}"
    preferred_dashboard = f"http://localhost:{DEFAULT_DASHBOARD_PORT}"
    existing_backend = url_is_ready(f"{preferred_api}/api/health")
    existing_dashboard = url_is_ready(f"{preferred_dashboard}/")
    if existing_backend or existing_dashboard:
        if not (existing_backend and existing_dashboard):
            occupied = "FastAPI" if existing_backend else "Dashboard"
            raise SystemExit(
                f"{occupied} cũ vẫn đang chạy nhưng dịch vụ còn lại không phản hồi. "
                "Hãy đóng cửa sổ DMS cũ rồi chạy lại để tránh lệch cổng."
            )
        if args.smoke_test:
            print("PASS smoke test: đang tái sử dụng FastAPI và dashboard tại cổng 8000/3000.")
            return
        if args.no_edge_launcher:
            print("DMS đã chạy tại http://localhost:3000 và Edge Launcher đã bị tắt theo tùy chọn.")
            return
        launcher_env = os.environ.copy()
        launcher_env["DMS_API_BASE"] = preferred_api
        launcher_env["DMS_DASHBOARD_URL"] = preferred_dashboard
        print("DMS đã chạy tại cổng 8000/3000; đang mở lại Edge Demo Launcher.")
        edge_launcher = subprocess.Popen(
            [sys.executable, "-B", str(EDGE_LAUNCHER)], cwd=PROJECT_ROOT, env=launcher_env
        )
        try:
            edge_launcher.wait()
        except KeyboardInterrupt:
            terminate(edge_launcher)
        return

    backend_port = find_free_port(DEFAULT_BACKEND_PORT)
    dashboard_port = find_free_port(DEFAULT_DASHBOARD_PORT)
    backend_host = "0.0.0.0" if args.lan else "127.0.0.1"
    env = os.environ.copy()
    if args.lan and (
        not env.get("DMS_JWT_SECRET")
        or env["DMS_JWT_SECRET"].startswith("development-only")
        or len(env["DMS_JWT_SECRET"]) < 32
    ):
        env["DMS_JWT_SECRET"] = secrets.token_urlsafe(48)
        print("Đã tạo JWT secret ngẫu nhiên cho phiên chạy LAN này.")
    env["VITE_API_BASE"] = f"http://127.0.0.1:{backend_port}"
    env["VITE_WS_URL"] = f"ws://127.0.0.1:{backend_port}/ws/monitor"

    backend = dashboard = edge_launcher = None
    try:
        backend = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", backend_host, "--port", str(backend_port)],
            cwd=PROJECT_ROOT,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if not wait_for_backend(backend_port):
            raise RuntimeError(f"Backend không sẵn sàng sau 30 giây trên cổng {backend_port}")
        dashboard = subprocess.Popen(
            [npm, "run", "dev", "--", "--host", "127.0.0.1", "--port", str(dashboard_port), "--strictPort"],
            cwd=DASHBOARD_ROOT,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        # vinext reports/binds localhost (often IPv6 ::1 on Windows) even when
        # the CLI receives 127.0.0.1, so probe the same hostname users open.
        if not wait_for_process_url(dashboard, f"http://localhost:{dashboard_port}/", 60.0):
            return_code = dashboard.poll()
            detail = f"; tiến trình đã dừng với mã {return_code}" if return_code is not None else ""
            raise RuntimeError(
                f"Dashboard không sẵn sàng sau 60 giây trên cổng {dashboard_port}{detail}"
            )

        if not args.smoke_test and not args.no_edge_launcher:
            launcher_env = env.copy()
            launcher_env["DMS_API_BASE"] = f"http://127.0.0.1:{backend_port}"
            launcher_env["DMS_DASHBOARD_URL"] = f"http://localhost:{dashboard_port}"
            edge_launcher = subprocess.Popen(
                [sys.executable, "-B", str(EDGE_LAUNCHER)],
                cwd=PROJECT_ROOT,
                env=launcher_env,
            )

        print("\nDMS đã khởi động:")
        print(f"  Dashboard: http://localhost:{dashboard_port}")
        print(f"  API docs:  http://127.0.0.1:{backend_port}/docs")
        if edge_launcher is not None:
            print("  Edge:      cửa sổ DMS Edge Demo Launcher đã được mở")
        if args.lan:
            print(f"  Mobile LAN API: http://<IPv4-máy-tính>:{backend_port}")
            print("  Chỉ dùng --lan trên Wi-Fi tin cậy; không mở cổng router ra Internet.")
        if args.smoke_test:
            if backend.poll() is not None or dashboard.poll() is not None:
                raise RuntimeError("Một tiến trình đã dừng trước khi hoàn tất smoke test")
            print("PASS smoke test: FastAPI, MySQL và dashboard đều phản hồi.")
            return
        print("Nhấn Ctrl+C để dừng cả hai dịch vụ.\n")

        while backend.poll() is None and dashboard.poll() is None:
            time.sleep(0.5)
        if backend.poll() is not None:
            raise RuntimeError(f"Backend đã dừng với mã {backend.returncode}")
        raise RuntimeError(f"Dashboard đã dừng với mã {dashboard.returncode}")
    except KeyboardInterrupt:
        print("\nĐang dừng DMS...")
    finally:
        terminate(edge_launcher)
        terminate(dashboard)
        terminate(backend)


if __name__ == "__main__":
    main()
