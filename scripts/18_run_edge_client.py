from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


# Tuyến demo khép kín quanh khu vực trung tâm TP.HCM. Đây là dữ liệu mô phỏng
# có chủ đích, không phải tọa độ GPS đo từ phần cứng.
SIMULATED_ROUTE = [
    (10.776889, 106.700806, 24.0, 90.0),
    (10.776975, 106.702103, 27.0, 88.0),
    (10.777126, 106.703411, 31.0, 85.0),
    (10.776402, 106.704096, 22.0, 175.0),
    (10.775514, 106.703901, 18.0, 260.0),
    (10.775365, 106.702551, 26.0, 270.0),
    (10.775803, 106.701201, 20.0, 315.0),
]


def post_json(url: str, payload: dict, headers: dict[str, str]) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="DMS Edge heartbeat/GPS client")
    parser.add_argument("--once", action="store_true", help="Gửi một heartbeat rồi dừng")
    parser.add_argument("--interval", type=float, default=20.0)
    parser.add_argument("--trip-id", type=int)
    parser.add_argument("--latitude", type=float)
    parser.add_argument("--longitude", type=float)
    parser.add_argument(
        "--simulate-gps",
        action="store_true",
        help="Gửi lần lượt tuyến GPS demo cố định; phải kèm --trip-id",
    )
    args = parser.parse_args()

    api_base = os.getenv("DMS_API_BASE", "http://127.0.0.1:8000").rstrip("/")
    device_id = os.getenv("DMS_EDGE_DEVICE_ID", "").strip()
    api_key = os.getenv("DMS_EDGE_API_KEY", "").strip()
    if not device_id or not api_key:
        raise SystemExit("Thiếu DMS_EDGE_DEVICE_ID hoặc DMS_EDGE_API_KEY")
    if (args.latitude is None) != (args.longitude is None):
        raise SystemExit("Phải cung cấp đồng thời --latitude và --longitude")
    if args.simulate_gps and not args.trip_id:
        raise SystemExit("--simulate-gps phải dùng cùng --trip-id")
    if args.simulate_gps and args.latitude is not None:
        raise SystemExit("Không dùng --latitude/--longitude cùng --simulate-gps")

    headers = {"X-Edge-Device-ID": device_id, "X-Edge-API-Key": api_key}
    print(f"Edge client: {device_id} -> {api_base}")
    route_index = 0
    while True:
        try:
            result = post_json(
                f"{api_base}/api/edge/heartbeat",
                {"software_version": "edge-simulator-1.0"}, headers,
            )
            print(f"Heartbeat OK: {result['server_time']}")
            if args.trip_id and (args.latitude is not None or args.simulate_gps):
                if args.simulate_gps:
                    latitude, longitude, speed_kph, heading = SIMULATED_ROUTE[route_index]
                    route_index = (route_index + 1) % len(SIMULATED_ROUTE)
                    source = "gps-simulated"
                else:
                    latitude, longitude = args.latitude, args.longitude
                    speed_kph = heading = None
                    source = "gps-manual"
                location = post_json(
                    f"{api_base}/api/edge/trips/{args.trip_id}/locations",
                    {
                        "latitude": latitude,
                        "longitude": longitude,
                        "speed_kph": speed_kph,
                        "heading": heading,
                        "source": source,
                    },
                    headers,
                )
                print(
                    f"GPS OK: id={location['location']['id']} "
                    f"({latitude:.6f}, {longitude:.6f}) [{source}]"
                )
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            print(f"HTTP {error.code}: {detail}")
        except OSError as error:
            print(f"Không kết nối được backend: {error}")
        if args.once:
            break
        time.sleep(max(args.interval, 5.0))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nĐã dừng Edge client.")
