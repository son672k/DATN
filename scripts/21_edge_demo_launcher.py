#!/usr/bin/env python3
"""Small desktop launcher for the Edge Simulator demo."""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_DIR = Path(os.getenv("LOCALAPPDATA", Path.home())) / "DMS_Edge_Simulator"
SETTINGS_FILE = SETTINGS_DIR / "launcher.json"


class Launcher:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.process: subprocess.Popen | None = None
        self.close_requested = False
        self.output_queue: queue.Queue[str] = queue.Queue()
        self.trip_map: dict[str, int] = {}
        self.saved = self.load_settings()

        root.title("DMS Edge Demo Launcher")
        root.geometry("780x680")
        root.minsize(700, 600)
        root.configure(bg="#edf3f2")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.build_ui()
        self.root.after(100, self.drain_output)

    @staticmethod
    def load_settings() -> dict:
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def save_settings(self) -> None:
        payload = {
            "api_base": self.api_base.get().strip(),
            "device_uid": self.device_uid.get().strip(),
            # Đây là launcher cục bộ dành cho máy demo. Người dùng đã chọn
            # lưu key để có thể mở phiên Edge kế tiếp mà không phải tạo lại.
            "api_key": self.api_key.get().strip() if self.remember_key.get() else "",
            "remember_key": self.remember_key.get(),
            "source": self.source.get().strip(),
            "camera": self.use_camera.get(),
            "gps": self.simulate_gps.get(),
            "device": self.compute_device.get(),
        }
        SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#edf3f2")
        style.configure("Card.TFrame", background="white")
        style.configure("Title.TLabel", background="#edf3f2", foreground="#123b35", font=("Segoe UI", 20, "bold"))
        style.configure("Sub.TLabel", background="#edf3f2", foreground="#52706b", font=("Segoe UI", 10))
        style.configure("Card.TLabel", background="white", foreground="#183d38", font=("Segoe UI", 10))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(16, 9))
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 7))
        style.configure("TEntry", padding=7)
        style.configure("TCombobox", padding=6)

        outer = ttk.Frame(self.root, padding=24)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="DMS Edge Demo Launcher", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="Chọn chuyến và nguồn hình ảnh, sau đó bấm Bắt đầu. AI vẫn chạy trong tiến trình Edge riêng.",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(2, 16))

        card = ttk.Frame(outer, style="Card.TFrame", padding=18)
        card.pack(fill="x")
        card.columnconfigure(1, weight=1)

        self.api_base = tk.StringVar(value=self.saved.get("api_base", os.getenv("DMS_API_BASE", "http://127.0.0.1:8000")))
        self.device_uid = tk.StringVar(value=self.saved.get("device_uid", os.getenv("DMS_EDGE_DEVICE_ID", "edge-car-001")))
        self.api_key = tk.StringVar(value=os.getenv("DMS_EDGE_API_KEY", self.saved.get("api_key", "")))
        self.show_api_key = tk.BooleanVar(value=True)
        self.remember_key = tk.BooleanVar(value=bool(self.saved.get("remember_key", True)))
        self.trip = tk.StringVar()
        self.source = tk.StringVar(value=self.saved.get("source", str(PROJECT_ROOT / "data" / "test_videos" / "test_video.mp4")))
        self.use_camera = tk.BooleanVar(value=bool(self.saved.get("camera", False)))
        self.simulate_gps = tk.BooleanVar(value=bool(self.saved.get("gps", True)))
        self.compute_device = tk.StringVar(value=self.saved.get("device", "auto"))
        self.replace_active_session = tk.BooleanVar(value=False)

        def row(label: str, widget, index: int, extra=None):
            ttk.Label(card, text=label, style="Card.TLabel").grid(row=index, column=0, sticky="w", padx=(0, 12), pady=6)
            widget.grid(row=index, column=1, sticky="ew", pady=6)
            if extra is not None:
                extra.grid(row=index, column=2, padx=(8, 0), pady=6)

        row("FastAPI", ttk.Entry(card, textvariable=self.api_base), 0)
        row("Mã thiết bị Edge", ttk.Entry(card, textvariable=self.device_uid), 1)
        self.api_key_entry = ttk.Entry(card, textvariable=self.api_key, show="")
        self.key_visibility_button = ttk.Button(card, text="Ẩn key", command=self.toggle_api_key)
        row("API key", self.api_key_entry, 2, self.key_visibility_button)
        self.trip_box = ttk.Combobox(card, textvariable=self.trip, state="readonly")
        row("Chuyến đang chạy", self.trip_box, 3, ttk.Button(card, text="Tải chuyến", command=self.load_trips))
        self.source_entry = ttk.Entry(card, textvariable=self.source)
        row("Video", self.source_entry, 4, ttk.Button(card, text="Chọn tệp", command=self.pick_video))
        row("Thiết bị tính toán", ttk.Combobox(card, textvariable=self.compute_device, values=("auto", "cpu", "cuda"), state="readonly"), 5)

        options = ttk.Frame(card, style="Card.TFrame")
        options.grid(row=6, column=0, columnspan=3, sticky="w", pady=(10, 3))
        ttk.Checkbutton(options, text="Dùng webcam camera:0", variable=self.use_camera, command=self.toggle_source).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(options, text="Mô phỏng GPS", variable=self.simulate_gps).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(options, text="Ghi nhớ API key", variable=self.remember_key).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(
            options,
            text="Kết thúc phiên cũ của thiết bị",
            variable=self.replace_active_session,
        ).pack(side="left")

        actions = ttk.Frame(outer, padding=(0, 14))
        actions.pack(fill="x")
        self.start_button = ttk.Button(actions, text="▶  Bắt đầu Edge", style="Primary.TButton", command=self.start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(actions, text="■  Dừng", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        ttk.Button(actions, text="🔊  Thử giọng Việt", command=self.test_audio).pack(side="left")
        ttk.Button(actions, text="Mở Dashboard", command=self.open_dashboard).pack(side="right")

        ttk.Label(outer, text="Nhật ký chạy", style="Sub.TLabel").pack(anchor="w", pady=(2, 5))
        self.log = tk.Text(outer, height=13, bg="#102724", fg="#d9f4ed", insertbackground="white", font=("Consolas", 9), relief="flat", padx=12, pady=10)
        self.log.pack(fill="both", expand=True)
        self.write("Sẵn sàng. Hãy bấm Tải chuyến để kiểm tra kết nối FastAPI.\n")
        self.toggle_source()
        if self.api_key.get().strip():
            self.root.after(600, self.load_trips)

    def headers(self) -> dict[str, str]:
        return {"X-Edge-Device-ID": self.device_uid.get().strip(), "X-Edge-API-Key": self.api_key.get().strip()}

    def load_trips(self) -> None:
        if not self.device_uid.get().strip() or not self.api_key.get().strip():
            messagebox.showwarning("Thiếu thông tin", "Hãy nhập mã thiết bị Edge và API key.")
            return
        request = urllib.request.Request(f"{self.api_base.get().rstrip('/')}/api/edge/trips/running", headers=self.headers())
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                trips = json.loads(response.read().decode("utf-8")).get("trips", [])
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            messagebox.showerror("Không tải được chuyến", f"HTTP {error.code}: {detail}")
            return
        except OSError as error:
            messagebox.showerror("Không kết nối được", str(error))
            return
        self.trip_map = {
            f"#{item['id']} · {item['trip_code']} · {item.get('driver_name') or 'Tài xế'} · {item.get('plate_number') or 'Xe'}": int(item["id"])
            for item in trips
        }
        values = list(self.trip_map)
        self.trip_box.configure(values=values)
        if values:
            self.trip.set(values[0])
            self.write(f"Đã tìm thấy {len(values)} chuyến đang chạy của thiết bị.\n")
        else:
            self.trip.set("")
            self.write(
                "Không có chuyến đang chạy gắn với xe của thiết bị này. "
                "Hãy kiểm tra mã thiết bị, xe được gắn với thiết bị và trạng thái chuyến trên Dashboard.\n"
            )

    def pick_video(self) -> None:
        selected = filedialog.askopenfilename(
            title="Chọn video cabin", filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("Tất cả tệp", "*.*")]
        )
        if selected:
            self.source.set(selected)

    def toggle_source(self) -> None:
        self.source_entry.configure(state="disabled" if self.use_camera.get() else "normal")

    def toggle_api_key(self) -> None:
        visible = not self.show_api_key.get()
        self.show_api_key.set(visible)
        self.api_key_entry.configure(show="" if visible else "•")
        self.key_visibility_button.configure(text="Ẩn key" if visible else "Hiện key")

    def test_audio(self) -> None:
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from backend.app.services.audio_alert_service import audio_backend, play_vehicle_alert

            backend = audio_backend(["distraction"])
            play_vehicle_alert("warning", ["distraction"])
            self.write(f"Đang thử cảnh báo bằng {backend}.\n")
        except (ImportError, OSError, RuntimeError) as error:
            messagebox.showerror("Không phát được âm thanh", str(error))

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        if not self.device_uid.get().strip() or not self.api_key.get().strip():
            messagebox.showwarning(
                "Thiếu thông tin Edge",
                "Hãy nhập mã thiết bị và API key. Có thể lấy key trong mục Tài xế & thiết bị trên Dashboard.",
            )
            return
        if self.trip.get() not in self.trip_map:
            self.load_trips()
            if self.trip.get() not in self.trip_map:
                messagebox.showwarning(
                    "Không có chuyến phù hợp",
                    "Thiết bị này chưa có chuyến đang chạy trên đúng phương tiện. "
                    "Trên Dashboard, hãy tạo hoặc chọn chuyến của xe gắn với thiết bị rồi bấm Bắt đầu.",
                )
                return
        if not self.use_camera.get() and not Path(self.source.get()).is_file():
            messagebox.showwarning("Thiếu video", "Hãy chọn một tệp video hợp lệ.")
            return
        if self.replace_active_session.get() and not messagebox.askyesno(
            "Thay thế phiên đang chạy",
            "Nếu thiết bị đang có phiên hoạt động, phiên đó sẽ được đánh dấu bị gián đoạn. Tiếp tục?",
        ):
            return
        self.save_settings()
        command = [
            sys.executable, "-B", str(PROJECT_ROOT / "scripts" / "20_run_edge_simulator.py"),
            "--trip-id", str(self.trip_map[self.trip.get()]),
            "--source", "camera:0" if self.use_camera.get() else self.source.get(),
        ]
        if self.replace_active_session.get():
            command.append("--replace-active-session")
        if self.simulate_gps.get():
            command.append("--simulate-gps")
        if self.compute_device.get() != "auto":
            command.extend(["--device", self.compute_device.get()])
        env = os.environ.copy()
        env.update({
            "DMS_API_BASE": self.api_base.get().strip(),
            "DMS_EDGE_DEVICE_ID": self.device_uid.get().strip(),
            "DMS_EDGE_API_KEY": self.api_key.get().strip(),
        })
        flags = 0
        if os.name == "nt":
            flags = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
        try:
            self.process = subprocess.Popen(
                command, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=flags,
            )
        except OSError as error:
            messagebox.showerror("Không thể chạy Edge", str(error))
            return
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.write("\n=== BẮT ĐẦU PHIÊN EDGE ===\n")
        threading.Thread(target=self.read_process, daemon=True).start()

    def read_process(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            self.output_queue.put(line)
        code = self.process.wait()
        self.output_queue.put(f"\n=== EDGE ĐÃ DỪNG · MÃ {code} ===\n")

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            if os.name == "nt":
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.process.send_signal(signal.SIGINT)
            self.write("Đang yêu cầu Edge kết thúc an toàn...\n")
        except OSError:
            self.process.terminate()

    def process_finished(self) -> None:
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    def drain_output(self) -> None:
        while True:
            try:
                self.write(self.output_queue.get_nowait())
            except queue.Empty:
                break
        if self.process is not None and self.process.poll() is not None:
            self.process_finished()
            if self.close_requested:
                self.root.destroy()
                return
        self.root.after(100, self.drain_output)

    def write(self, text: str) -> None:
        self.log.insert("end", text)
        self.log.see("end")

    def open_dashboard(self) -> None:
        import webbrowser
        webbrowser.open(os.getenv("DMS_DASHBOARD_URL", "http://localhost:3000"))

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno("Edge đang chạy", "Dừng Edge và đóng cửa sổ?"):
                return
            self.close_requested = True
            self.stop()
            return
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    Launcher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
