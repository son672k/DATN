from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

LOCAL = timezone(timedelta(hours=7))
PENALTIES = {"drowsy": 10, "phone": 8, "cigarette": 5, "distraction": 5}

def stamp(value):
    if not value:
        return None
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result
    except ValueError:
        return None

def window(period="month", day=None):
    current = datetime.fromisoformat(day).replace(tzinfo=LOCAL) if day else datetime.now(LOCAL)
    start = current.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "week":
        start -= timedelta(days=start.weekday())
        end = start + timedelta(days=7)
    elif period == "month":
        start = start.replace(day=1)
        end = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    else:
        end = start + timedelta(days=1)
    return start, end

def aggregate(sessions, events, period="month", day=None):
    start, end = window(period, day)
    selected = [e for e in events if (t := stamp(e.get("occurred_at"))) and start <= t < end]
    durations = defaultdict(float)
    nights = defaultdict(float)
    for s in sessions:
        t = stamp(s.get("started_at"))
        if not t:
            continue
        # Monitoring duration is not certified driving time. Clip to report period.
        finish = t + timedelta(seconds=max(0, float(s.get("video_duration_seconds") or 0)))
        cursor, finish = max(t, start), min(finish, end)
        while cursor < finish:
            local = cursor.astimezone(LOCAL)
            boundary = local.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            stop = min(boundary, finish)
            seconds = (stop - cursor).total_seconds()
            durations[s.get("driver_id")] += seconds
            if local.hour >= 22 or local.hour < 6:
                nights[s.get("driver_id")] += seconds
            cursor = stop
    # Coalesce overlapping fatigue signals in the same session within 30 seconds.
    episodes = []
    last = {}
    for e in sorted(selected, key=lambda x: (x["session_id"], float(x["timestamp_seconds"]), x["id"])):
        kind = "drowsy" if e["event_type"] in {"drowsy", "microsleep", "high_perclos"} else e["event_type"]
        if kind not in PENALTIES:
            continue
        key = (e["session_id"], kind)
        t = float(e["timestamp_seconds"])
        if key not in last or t - last[key] >= 30:
            episodes.append((e, PENALTIES[kind]))
            last[key] = t
    names = {s.get("driver_id"): s.get("driver_name") for s in sessions}
    rows = []
    for driver in set(durations) | {e.get("driver_id") for e in selected}:
        if driver is None:
            continue
        own = [e for e in selected if e.get("driver_id") == driver]
        deduction = sum(p for e, p in episodes if e.get("driver_id") == driver)
        score = max(0, 100 - deduction)
        rows.append({"driver_id": driver, "driver_name": names.get(driver) or str(driver),
                     "safety_score": score, "deduction": deduction,
                     "rating": "Xuất sắc" if score >= 90 else "Tốt" if score >= 80 else "Trung bình" if score >= 70 else "Nguy cơ cao",
                     "monitoring_hours": round(durations[driver] / 3600, 3),
                     "night_monitoring_hours": round(nights[driver] / 3600, 3),
                     "total_events": len(own), "event_counts": dict(Counter(e["event_type"] for e in own))})
    return {"period": period, "start": start.isoformat(), "end": end.isoformat(),
            "policy": "Điểm tham khảo trong kỳ: 100 trừ ngủ gật 10, điện thoại 8, thuốc lá 5, mất tập trung 5. Gộp tín hiệu cùng nhóm trong 30 giây/phiên. Giờ đêm 22–06. Thời gian là thời gian giám sát; không phải xác suất tai nạn.",
            "drivers": sorted(rows, key=lambda x: (-x["safety_score"], x["driver_id"])),
            "top_violations": sorted(rows, key=lambda x: (-x["total_events"], x["driver_id"])),
            "event_counts": dict(Counter(e["event_type"] for e in selected)),
            "hour_counts": dict(sorted(Counter(stamp(e["occurred_at"]).astimezone(LOCAL).hour for e in selected).items())),
            "route_counts": dict(Counter(e.get("route_name") or "Chưa gán tuyến" for e in selected)),
            "total_events": len(selected)}
