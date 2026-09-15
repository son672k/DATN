from __future__ import annotations

import asyncio
import csv
import io
import queue
import shutil
import threading
import time
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, Request, Response, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from ..config import settings
from ..database import Database, utc_now
from ..services.audio_alert_service import ALERT_MESSAGES
from ..security import (
    create_access_token, decode_access_token, generate_edge_api_key,
    hash_edge_api_key, hash_password, verify_edge_api_key, verify_password,
)
from ..schemas import (
    AdminDriverAccountRequest,
    DriverProfileUpdateRequest,
    EdgeDeviceCreateRequest,
    EdgeDeviceUpdateRequest,
    EdgeEventCreateRequest,
    EdgeHeartbeatRequest,
    EdgeMetricBatchRequest,
    EdgeSessionCompleteRequest,
    EdgeSessionCreateRequest,
    EventReviewRequest,
    PasswordChangeRequest,
    PasswordResetRequest,
    RouteCreateRequest,
    RouteUpdateRequest,
    StartSessionRequest,
    StartSessionResponse,
    TripCreateRequest,
    TripUpdateRequest,
    TripStatusRequest,
    UserActiveRequest,
    VehicleCreateRequest,
    VehicleLocationRequest,
    VehicleUpdateRequest,
)
from ..services.inference_service import InferenceManager
from ..services.report_service import build_session_report
from ..services.safety_service import aggregate


def create_router(database: Database, manager: InferenceManager) -> APIRouter:
    router = APIRouter()
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")
    allowed_video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    max_upload_bytes = 500 * 1024 * 1024
    login_attempts: dict[str, list[float]] = {}
    login_attempt_lock = threading.Lock()
    login_window_seconds = 300.0
    login_max_failures = 5
    login_attempt_capacity = 5000
    edge_session_locks: dict[int, threading.Lock] = {}
    edge_session_locks_guard = threading.Lock()

    def public_user(user: dict) -> dict:
        driver = (
            database.get_driver(int(user["driver_id"]))
            if user.get("driver_id") is not None
            else None
        )
        return {
            "id": user["id"],
            "username": user["username"],
            "display_name": user["display_name"],
            "role": user["role"],
            "driver_id": user.get("driver_id"),
            "active": bool(user["active"]),
            "driver": {
                "id": driver["id"],
                "external_id": driver["external_id"],
                "display_name": driver["display_name"],
            } if driver else None,
        }

    def authenticated_user_from_token(token: str) -> dict:
        try:
            payload = decode_access_token(token, settings.jwt_secret)
            user = database.get_user(int(payload["sub"]))
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError("Phiên đăng nhập không hợp lệ hoặc đã hết hạn") from error
        if (
            user is None
            or not bool(user["active"])
            or int(payload.get("ver", -1)) != int(user.get("token_version") or 0)
        ):
            raise ValueError("Phiên đăng nhập không hợp lệ hoặc đã hết hạn")
        return user

    def current_user(token: str = Depends(oauth2_scheme)) -> dict:
        try:
            return authenticated_user_from_token(token)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(error),
                headers={"WWW-Authenticate": "Bearer"},
            ) from error

    def require_supervisor(user: dict = Depends(current_user)) -> dict:
        if user["role"] != "supervisor":
            raise HTTPException(status_code=403, detail="Cần quyền người giám sát")
        return user

    def require_driver(user: dict = Depends(current_user)) -> dict:
        if user["role"] != "driver" or user.get("driver_id") is None:
            raise HTTPException(status_code=403, detail="Cần quyền tài xế")
        return user

    def ensure_event_access(event: dict | None, user: dict) -> dict:
        if event is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy sự kiện")
        if user["role"] == "supervisor":
            return event
        session = database.get_session(int(event["session_id"]))
        if (
            user["role"] != "driver"
            or user.get("driver_id") is None
            or session is None
            or int(session.get("driver_id") or 0) != int(user["driver_id"])
        ):
            raise HTTPException(status_code=403, detail="Không được xem dữ liệu của tài xế khác")
        return event

    def require_edge_device(
        device_uid: str = Header(alias="X-Edge-Device-ID"),
        api_key: str = Header(alias="X-Edge-API-Key"),
    ) -> dict:
        device = database.get_edge_device_by_uid(device_uid)
        if (
            device is None
            or not bool(device["active"])
            or not verify_edge_api_key(api_key, device["api_key_hash"])
        ):
            raise HTTPException(status_code=401, detail="Thông tin xác thực Edge không hợp lệ")
        return device

    def require_edge_session(session_id: int, device: dict) -> dict:
        session = database.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy phiên Edge")
        if int(session.get("edge_device_id") or 0) != int(device["id"]):
            raise HTTPException(status_code=403, detail="Phiên không thuộc thiết bị Edge này")
        return session

    def publish_edge(message: dict) -> None:
        publisher = getattr(manager, "publish_edge_message", None)
        if callable(publisher):
            publisher(message)

    @router.get("/api/health")
    def health():
        active_session = database.get_active_session()
        return {
            "status": "ok",
            "database": "mysql" if database.is_mysql else "test-sqlite",
            "active_session_id": manager.active_session_id or (active_session["id"] if active_session else None),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "realtime_defaults": {
                "yolo_interval": 2,
                "cnn_interval": 2,
                "save_video": False,
            },
        }

    @router.get("/api/models/status")
    def model_status(_: dict = Depends(require_supervisor)):
        models = {
            "yolo": settings.yolo_checkpoint,
            "cnn": settings.cnn_checkpoint,
            "lstm": settings.lstm_checkpoint,
            "face_detector": settings.project_root / "models" / "blaze_face_short_range.tflite",
            "face_landmarker": settings.face_landmarker,
        }
        return {
            "ready": all(path.is_file() for path in models.values()),
            "models": {
                name: {
                    "ready": path.is_file(),
                    "filename": path.name,
                    "size_bytes": path.stat().st_size if path.is_file() else 0,
                }
                for name, path in models.items()
            },
        }

    @router.post("/api/auth/token")
    def login(request: Request, form: OAuth2PasswordRequestForm = Depends()):
        username = form.username.strip().lower()
        client_host = request.client.host if request.client else "unknown"
        attempt_key = f"{client_host}:{username}"
        now = time.monotonic()
        with login_attempt_lock:
            for key, attempts in list(login_attempts.items()):
                fresh = [value for value in attempts if now - value < login_window_seconds]
                if fresh:
                    login_attempts[key] = fresh
                else:
                    login_attempts.pop(key, None)
            if attempt_key not in login_attempts and len(login_attempts) >= login_attempt_capacity:
                oldest_key = min(login_attempts, key=lambda key: login_attempts[key][-1])
                login_attempts.pop(oldest_key, None)
            recent = [
                attempted for attempted in login_attempts.get(attempt_key, [])
                if now - attempted < login_window_seconds
            ]
            login_attempts[attempt_key] = recent
        if len(recent) >= login_max_failures:
            raise HTTPException(
                status_code=429,
                detail="Đăng nhập sai quá nhiều lần; hãy thử lại sau 5 phút",
                headers={"Retry-After": str(int(login_window_seconds))},
            )
        user = database.get_user_by_username(username)
        if user is None or not bool(user["active"]) or not verify_password(
            form.password, user["password_hash"]
        ):
            with login_attempt_lock:
                login_attempts.setdefault(attempt_key, []).append(now)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Sai tên đăng nhập hoặc mật khẩu",
                headers={"WWW-Authenticate": "Bearer"},
            )
        with login_attempt_lock:
            login_attempts.pop(attempt_key, None)
        return {
            "access_token": create_access_token(
                user, settings.jwt_secret, settings.jwt_expire_minutes
            ),
            "token_type": "bearer",
            "expires_in": settings.jwt_expire_minutes * 60,
            "user": public_user(user),
        }

    @router.get("/api/auth/me")
    def auth_me(user: dict = Depends(current_user)):
        return {"user": public_user(user)}

    @router.get("/api/driver/me/dashboard")
    def driver_own_dashboard(user: dict = Depends(require_driver)):
        driver_id = int(user["driver_id"])
        driver = database.get_driver(driver_id)
        sessions = database.list_driver_sessions(driver_id, 100)
        events = database.list_driver_recent_events(driver_id, 100)
        return {
            "driver": driver,
            "sessions": sessions,
            "alerts": events,
            "trips": database.list_driver_trips(driver_id),
            "event_counts": database.driver_event_counts(driver_id),
            "summary": {
                "total_sessions": len(sessions),
                "total_events": len(events),
                "max_risk_score": max((int(row.get("risk_score") or 0) for row in events), default=0),
            },
        }

    @router.get("/api/driver/me/alerts")
    def driver_own_alerts(
        limit: int = Query(default=100, ge=1, le=500),
        user: dict = Depends(require_driver),
    ):
        return {"alerts": database.list_driver_recent_events(int(user["driver_id"]), limit)}

    @router.get("/api/driver/me/trips")
    def driver_own_trips(user: dict = Depends(require_driver)):
        return {"trips": database.list_driver_trips(int(user["driver_id"]))}

    @router.get("/api/driver/me/trips/{trip_id}/locations")
    def driver_own_trip_locations(
        trip_id: int,
        limit: int = Query(default=500, ge=1, le=5000),
        user: dict = Depends(require_driver),
    ):
        if database.get_driver_trip(int(user["driver_id"]), trip_id) is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy chuyến đi của tài xế")
        return {"locations": database.list_trip_locations(trip_id, limit), "events": database.trip_events(trip_id)}

    @router.post("/api/auth/change-password")
    def change_password(
        payload: PasswordChangeRequest,
        user: dict = Depends(current_user),
    ):
        if not verify_password(payload.current_password, user["password_hash"]):
            raise HTTPException(status_code=400, detail="Mật khẩu hiện tại không đúng")
        if payload.current_password == payload.new_password:
            raise HTTPException(status_code=400, detail="Mật khẩu mới phải khác mật khẩu hiện tại")
        updated = database.update_user_password(
            int(user["id"]), hash_password(payload.new_password)
        )
        assert updated is not None
        return {"changed": True, "message": "Đã đổi mật khẩu; hãy đăng nhập lại"}

    @router.get("/api/admin/users")
    def admin_users(_: dict = Depends(require_supervisor)):
        return {"users": database.list_users()}

    @router.get("/api/admin/drivers")
    def admin_drivers(_: dict = Depends(require_supervisor)):
        return {"drivers": database.list_drivers()}

    @router.get("/api/admin/edge-devices")
    def admin_edge_devices(_: dict = Depends(require_supervisor)):
        database.mark_stale_edge_devices_offline()
        return {"devices": database.list_edge_devices()}

    @router.post("/api/admin/edge-devices", status_code=201)
    def create_admin_edge_device(
        payload: EdgeDeviceCreateRequest,
        _: dict = Depends(require_supervisor),
    ):
        if database.get_edge_device_by_uid(payload.device_uid) is not None:
            raise HTTPException(status_code=409, detail="Mã thiết bị Edge đã tồn tại")
        if payload.vehicle_id is not None and database.get_vehicle(payload.vehicle_id) is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy phương tiện")
        api_key = generate_edge_api_key()
        device = database.create_edge_device(
            payload.device_uid, payload.label.strip(), hash_edge_api_key(api_key),
            payload.vehicle_id,
        )
        device.pop("api_key_hash", None)
        return {"device": device, "api_key": api_key}

    @router.post("/api/admin/driver-accounts", status_code=201)
    def create_driver_account(
        payload: AdminDriverAccountRequest,
        _: dict = Depends(require_supervisor),
    ):
        username = payload.username.strip().lower()
        if database.get_user_by_username(username) is not None:
            raise HTTPException(status_code=409, detail="Tên đăng nhập đã tồn tại")
        existing_driver = database.get_driver_by_external_id(payload.external_id.strip())
        if existing_driver is not None and database.get_user_by_driver_id(int(existing_driver["id"])):
            raise HTTPException(status_code=409, detail="Mã tài xế đã có tài khoản đăng nhập")
        driver = database.create_or_update_driver(
            payload.external_id.strip(), payload.display_name.strip(),
            payload.phone.strip() if payload.phone else None,
        )
        user = database.create_or_update_user(
            username, hash_password(payload.password), payload.display_name.strip(),
            "driver", int(driver["id"]),
        )
        return {"driver": driver, "user": public_user(user)}

    @router.get("/api/admin/fleet/overview")
    def admin_fleet_overview(_: dict = Depends(require_supervisor)):
        accounts = {row.get("driver_id"): row for row in database.list_users() if row.get("driver_id")}
        rows = []
        for driver in database.list_drivers():
            driver_id = int(driver["id"])
            driver_sessions = database.list_driver_sessions(driver_id, 100)
            latest_session = driver_sessions[0] if driver_sessions else None
            review_counts = database.driver_review_counts(driver_id)
            rows.append({
                "driver": driver,
                "account_active": bool(accounts.get(driver_id, {}).get("active", False)),
                "device_count": 0,
                "active_device_count": 0,
                "last_seen_at": None,
                "latest_session": latest_session,
                "total_sessions": len(driver_sessions),
                "review_counts": review_counts,
            })
        return {"drivers": rows}

    @router.get("/api/sessions/{session_id}/report.pdf")
    def session_pdf_report(session_id: int, _: dict = Depends(require_supervisor)):
        from ..services.pdf_report_service import session_pdf
        if database.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy phiên")
        return Response(session_pdf(database, session_id, settings.project_root), media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="dms_session_{session_id}.pdf"'})

    @router.get("/api/admin/reports/daily.pdf")
    def daily_pdf_report(day: str, _: dict = Depends(require_supervisor)):
        from ..services.pdf_report_service import daily_pdf
        from datetime import date
        try:
            day = date.fromisoformat(day).isoformat()
        except ValueError:
            raise HTTPException(status_code=422, detail="Ngày không hợp lệ; dùng YYYY-MM-DD")
        return Response(daily_pdf(database, day, settings.project_root), media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="dms_{day}.pdf"'})

    @router.get("/api/admin/analytics")
    def safety_analytics(
        period: str = Query(default="month", pattern="^(day|week|month)$"),
        day: str | None = None, _: dict = Depends(require_supervisor),
    ):
        try:
            return aggregate(*database.safety_records(), period, day)
        except ValueError:
            raise HTTPException(status_code=422, detail="Ngày không hợp lệ; dùng YYYY-MM-DD")

    @router.get("/api/admin/alerts")
    def admin_alert_queue(
        limit: int = Query(default=100, ge=1, le=500),
        status: str | None = Query(default=None, pattern=r"^(new|acknowledged|resolved)$"),
        _: dict = Depends(require_supervisor),
    ):
        return {"alerts": database.list_supervisor_alerts(limit, status)}

    @router.get("/api/admin/drivers/{driver_id}/dashboard")
    def admin_driver_dashboard(
        driver_id: int,
        _: dict = Depends(require_supervisor),
    ):
        driver = database.get_driver(driver_id)
        if driver is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy tài xế")
        sessions = database.list_driver_sessions(driver_id, 100)
        recent_events = database.list_driver_recent_events(driver_id, 30)
        event_counts = database.driver_event_counts(driver_id)
        frame_weight = sum(max(int(session["frame_count"] or 0), 1) for session in sessions)
        weighted_risk = sum(
            float(session["avg_risk_score"] or 0) * max(int(session["frame_count"] or 0), 1)
            for session in sessions
        )
        account = database.get_user_by_driver_id(driver_id)
        return {
            "driver": driver,
            "account": public_user(account) | {"active": bool(account["active"])} if account else None,
            "devices": [],
            "sessions": sessions,
            "recent_events": recent_events,
            "event_counts": event_counts,
            "summary": {
                "total_sessions": len(sessions),
                "completed_sessions": sum(session["status"] == "completed" for session in sessions),
                "total_frames": sum(int(session["frame_count"] or 0) for session in sessions),
                "total_duration_seconds": sum(float(session["video_duration_seconds"] or 0) for session in sessions),
                "avg_risk_score": weighted_risk / frame_weight if sessions else 0.0,
                "max_risk_score": max((int(session["max_risk_score"] or 0) for session in sessions), default=0),
                "total_events": sum(event_counts.values()),
            },
        }

    @router.patch("/api/admin/drivers/{driver_id}")
    def update_admin_driver(
        driver_id: int,
        payload: DriverProfileUpdateRequest,
        _: dict = Depends(require_supervisor),
    ):
        driver = database.update_driver_profile(
            driver_id, payload.display_name.strip(),
            payload.phone.strip() if payload.phone else None,
        )
        if driver is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy tài xế")
        return {"driver": driver}

    @router.patch("/api/admin/users/{user_id}/active")
    def set_admin_user_active(
        user_id: int,
        payload: UserActiveRequest,
        supervisor: dict = Depends(require_supervisor),
    ):
        target = database.get_user(user_id)
        if target is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy tài khoản")
        if target["role"] != "driver":
            raise HTTPException(status_code=403, detail="Chỉ được khóa hoặc mở tài khoản tài xế")
        if int(target["id"]) == int(supervisor["id"]) and not payload.active:
            raise HTTPException(status_code=409, detail="Không thể tự khóa tài khoản đang đăng nhập")
        updated = database.set_user_active(user_id, payload.active)
        assert updated is not None
        return {"user": public_user(updated)}

    @router.patch("/api/admin/users/{user_id}/password")
    def reset_admin_user_password(
        user_id: int,
        payload: PasswordResetRequest,
        _: dict = Depends(require_supervisor),
    ):
        target = database.get_user(user_id)
        if target is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy tài khoản")
        if target["role"] != "driver":
            raise HTTPException(status_code=403, detail="Chỉ được đặt lại mật khẩu tài xế")
        updated = database.update_user_password(user_id, hash_password(payload.new_password))
        assert updated is not None
        return {"reset": True, "user": public_user(updated)}

    @router.patch("/api/admin/edge-devices/{device_id}/active")
    def set_admin_edge_device_active(
        device_id: int,
        payload: UserActiveRequest,
        _: dict = Depends(require_supervisor),
    ):
        device = database.set_edge_device_active(device_id, payload.active)
        if device is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy thiết bị")
        return {"device": device}

    @router.patch("/api/admin/edge-devices/{device_id}")
    def update_admin_edge_device(
        device_id: int,
        payload: EdgeDeviceUpdateRequest,
        _: dict = Depends(require_supervisor),
    ):
        if payload.vehicle_id is not None and database.get_vehicle(payload.vehicle_id) is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy phương tiện")
        device = database.update_edge_device(
            device_id, payload.label.strip(), payload.vehicle_id,
        )
        if device is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy thiết bị Edge")
        return {"device": device}

    @router.delete("/api/admin/edge-devices/{device_id}")
    def delete_admin_edge_device(
        device_id: int,
        _: dict = Depends(require_supervisor),
    ):
        device = database.get_edge_device(device_id)
        if device is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy thiết bị Edge")
        if bool(device["active"]):
            raise HTTPException(status_code=409, detail="Hãy thu hồi thiết bị Edge trước khi xóa")
        database.delete_edge_device(device_id)
        return {"deleted": True, "device_id": device_id}

    @router.post("/api/admin/edge-devices/{device_id}/rotate-key")
    def rotate_admin_edge_device_key(
        device_id: int,
        _: dict = Depends(require_supervisor),
    ):
        if database.get_edge_device(device_id) is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy thiết bị Edge")
        api_key = generate_edge_api_key()
        device = database.rotate_edge_device_key(device_id, hash_edge_api_key(api_key))
        return {"device": device, "api_key": api_key}

    @router.post("/api/edge/heartbeat")
    def edge_heartbeat(
        _: EdgeHeartbeatRequest,
        device: dict = Depends(require_edge_device),
    ):
        updated = database.touch_edge_device(int(device["id"]))
        if updated is None:
            raise HTTPException(status_code=403, detail="Thiết bị Edge đã bị vô hiệu hóa")
        return {"status": "online", "device_uid": updated["device_uid"], "server_time": updated["last_seen_at"]}

    @router.get("/api/edge/trips/running")
    def edge_running_trips(device: dict = Depends(require_edge_device)):
        vehicle_id = device.get("vehicle_id")
        if vehicle_id is None:
            return {"trips": []}
        trips = [
            trip for trip in database.list_trips()
            if trip["status"] == "running" and int(trip["vehicle_id"]) == int(vehicle_id)
        ]
        return {"trips": trips}

    @router.post("/api/edge/sessions", status_code=201)
    def create_edge_session(
        payload: EdgeSessionCreateRequest,
        device: dict = Depends(require_edge_device),
    ):
        trip = database.get_trip(payload.trip_id)
        if trip is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy chuyến đi")
        if trip["status"] != "running":
            raise HTTPException(status_code=409, detail="Chuyến phải ở trạng thái đang chạy")
        if device.get("vehicle_id") is None or int(device["vehicle_id"]) != int(trip["vehicle_id"]):
            raise HTTPException(status_code=409, detail="Thiết bị Edge không thuộc phương tiện của chuyến")
        device_id = int(device["id"])
        with edge_session_locks_guard:
            device_lock = edge_session_locks.setdefault(device_id, threading.Lock())
        with device_lock:
            active = database.get_active_edge_session(device_id)
            if active is not None:
                if not payload.replace_active:
                    raise HTTPException(status_code=409, detail=f"Thiết bị đang có phiên #{active['id']}")
                database.update_session(
                    int(active["id"]),
                    status="interrupted",
                    ended_at=utc_now(),
                    error_message="Phiên Edge bị thay thế khi thiết bị bắt đầu phiên mới",
                )
                interrupted = database.get_session(int(active["id"]))
                publish_edge({
                    "type": "session_interrupted",
                    "session_id": int(active["id"]),
                    "session": interrupted,
                })
            session_id = database.create_session(
                payload.source, payload.device, payload.yolo_interval, payload.cnn_interval,
                payload.save_video, int(trip["id"]), device_id,
            )
            database.update_session(session_id, status="running", started_at=utc_now())
        database.touch_edge_device(int(device["id"]))
        session = database.get_session(session_id)
        publish_edge({"type": "edge_session_started", "session_id": session_id, "session": session})
        return {"session": session}

    @router.post("/api/edge/sessions/{session_id}/metrics")
    def add_edge_metrics(
        session_id: int,
        payload: EdgeMetricBatchRequest,
        device: dict = Depends(require_edge_device),
    ):
        session = require_edge_session(session_id, device)
        if session["status"] != "running":
            raise HTTPException(status_code=409, detail="Phiên Edge không còn hoạt động")
        required = {"frame_index", "timestamp_seconds", "risk_score", "severity", "warnings", "processing_fps"}
        if any(not required.issubset(sample) for sample in payload.samples):
            raise HTTPException(status_code=422, detail="Metric Edge thiếu trường bắt buộc")
        database.add_frame_metrics(session_id, payload.samples)
        frame_count = max(int(x["frame_index"]) for x in payload.samples) + 1
        database.update_session(session_id, frame_count=max(frame_count, int(session.get("frame_count") or 0)))
        database.touch_edge_device(int(device["id"]))
        if payload.live_message:
            message = {**payload.live_message, "type": "frame", "session_id": session_id}
            publish_edge(message)
        return {"accepted": len(payload.samples), "frame_count": frame_count}

    @router.post("/api/edge/sessions/{session_id}/events", status_code=201)
    def add_edge_event(
        session_id: int,
        payload: EdgeEventCreateRequest,
        device: dict = Depends(require_edge_device),
    ):
        session = require_edge_session(session_id, device)
        if session["status"] != "running":
            raise HTTPException(status_code=409, detail="Phiên Edge không còn hoạt động")
        event_id = database.add_event(session_id, {
            "type": payload.event_type,
            "frame_index": payload.frame_index,
            "timestamp_seconds": payload.timestamp_seconds,
            "confidence": payload.confidence,
            "risk_score": payload.risk_score,
            "client_event_id": payload.client_event_id,
            "event_payload": payload.payload,
        })
        event = database.get_event(event_id)
        database.touch_edge_device(int(device["id"]))
        publish_edge({"type": "event", "session_id": session_id, "event": event})
        return {"event": event}

    async def store_edge_evidence(
        session_id: int, event_id: int, field: str, suffix: str, limit: int,
        file: UploadFile, device: dict,
    ) -> dict:
        require_edge_session(session_id, device)
        event = database.get_event(event_id)
        if event is None or int(event["session_id"]) != session_id:
            raise HTTPException(status_code=404, detail="Không tìm thấy sự kiện trong phiên")
        folder = "snapshots" if field == "snapshot_path" else "clips"
        target_dir = settings.session_output_root / str(session_id) / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"edge_{event_id}{suffix}"
        total = 0
        try:
            with target.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > limit:
                        raise HTTPException(status_code=413, detail="Tệp bằng chứng Edge vượt giới hạn")
                    output.write(chunk)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        finally:
            await file.close()
        if total == 0:
            target.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail="Tệp bằng chứng Edge rỗng")
        relative = str(target.relative_to(settings.project_root)).replace("\\", "/")
        updated = database.update_event_evidence(event_id, field, relative)
        return {"event": updated, "size_bytes": total}

    @router.post("/api/edge/sessions/{session_id}/events/{event_id}/snapshot")
    async def upload_edge_snapshot(
        session_id: int, event_id: int, file: UploadFile = File(...),
        device: dict = Depends(require_edge_device),
    ):
        if file.content_type not in {"image/jpeg", "image/jpg"}:
            raise HTTPException(status_code=415, detail="Snapshot phải là JPEG")
        return await store_edge_evidence(session_id, event_id, "snapshot_path", ".jpg", 10 * 1024 * 1024, file, device)

    @router.post("/api/edge/sessions/{session_id}/events/{event_id}/clip")
    async def upload_edge_clip(
        session_id: int, event_id: int, file: UploadFile = File(...),
        device: dict = Depends(require_edge_device),
    ):
        if file.content_type != "video/mp4":
            raise HTTPException(status_code=415, detail="Clip phải là MP4")
        return await store_edge_evidence(session_id, event_id, "clip_path", ".mp4", 150 * 1024 * 1024, file, device)

    @router.post("/api/edge/sessions/{session_id}/preview")
    async def upload_edge_preview(
        session_id: int, file: UploadFile = File(...),
        device: dict = Depends(require_edge_device),
    ):
        session = require_edge_session(session_id, device)
        if session["status"] != "running":
            raise HTTPException(status_code=409, detail="Phiên Edge không còn hoạt động")
        jpeg = await file.read(2 * 1024 * 1024 + 1)
        await file.close()
        if len(jpeg) > 2 * 1024 * 1024 or not jpeg.startswith(b"\xff\xd8"):
            raise HTTPException(status_code=422, detail="Preview JPEG không hợp lệ")
        setter = getattr(manager, "set_edge_preview", None)
        if callable(setter):
            setter(session_id, jpeg)
        return {"accepted": True, "size_bytes": len(jpeg)}

    @router.post("/api/edge/sessions/{session_id}/complete")
    def complete_edge_session(
        session_id: int,
        payload: EdgeSessionCompleteRequest,
        device: dict = Depends(require_edge_device),
    ):
        session = require_edge_session(session_id, device)
        if session["status"] not in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="Phiên Edge đã kết thúc")
        database.update_session(
            session_id, status=payload.status, ended_at=utc_now(),
            frame_count=payload.frame_count,
            video_duration_seconds=payload.video_duration_seconds,
            processing_seconds=payload.processing_seconds, avg_fps=payload.avg_fps,
            avg_risk_score=payload.avg_risk_score, max_risk_score=payload.max_risk_score,
            face_detection_rate=payload.face_detection_rate,
            error_message=payload.error_message,
        )
        database.touch_edge_device(int(device["id"]))
        completed = database.get_session(session_id)
        if database.is_mysql:
            try:
                from ..services.pdf_report_service import save_atomic, session_pdf
                output_dir = settings.session_output_root / str(session_id)
                output_dir.mkdir(parents=True, exist_ok=True)
                save_atomic(output_dir / "report.pdf", session_pdf(database, session_id, settings.project_root))
            except Exception:
                import logging
                logging.getLogger(__name__).exception("Không tạo được PDF tự động cho phiên Edge %s", session_id)
        message_type = "session_complete" if payload.status != "failed" else "session_failed"
        message = {"type": message_type, "session_id": session_id, **completed}
        if payload.status == "failed":
            message["error"] = completed.get("error_message") or "Lỗi không xác định tại Edge"
        publish_edge(message)
        return {"session": completed}

    @router.get("/api/admin/vehicles")
    def admin_vehicles(_: dict = Depends(require_supervisor)):
        return {"vehicles": database.list_vehicles()}

    @router.post("/api/admin/vehicles", status_code=201)
    def create_admin_vehicle(
        payload: VehicleCreateRequest,
        _: dict = Depends(require_supervisor),
    ):
        plate = payload.plate_number.strip().upper()
        if any(item["plate_number"] == plate for item in database.list_vehicles()):
            raise HTTPException(status_code=409, detail="Biển số xe đã tồn tại")
        return {"vehicle": database.create_vehicle(
            plate, payload.vehicle_type.strip(), payload.model.strip() if payload.model else None
        )}

    @router.patch("/api/admin/vehicles/{vehicle_id}")
    def update_admin_vehicle(
        vehicle_id: int, payload: VehicleUpdateRequest,
        _: dict = Depends(require_supervisor),
    ):
        plate = payload.plate_number.strip().upper()
        if any(
            int(item["id"]) != vehicle_id and item["plate_number"] == plate
            for item in database.list_vehicles()
        ):
            raise HTTPException(status_code=409, detail="Biển số xe đã tồn tại")
        vehicle = database.update_vehicle(
            vehicle_id, payload.vehicle_type.strip(),
            payload.model.strip() if payload.model else None,
            payload.status, payload.active, plate,
        )
        if vehicle is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy phương tiện")
        return {"vehicle": vehicle}

    @router.delete("/api/admin/vehicles/{vehicle_id}")
    def delete_admin_vehicle(
        vehicle_id: int,
        _: dict = Depends(require_supervisor),
    ):
        if database.get_vehicle(vehicle_id) is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy phương tiện")
        references = database.vehicle_reference_counts(vehicle_id)
        if sum(references.values()):
            raise HTTPException(
                status_code=409,
                detail="Không thể xóa xe đã có thiết bị hoặc chuyến. Hãy ngừng hoạt động xe để giữ lịch sử.",
            )
        database.delete_vehicle(vehicle_id)
        return {"deleted": True, "vehicle_id": vehicle_id}

    @router.get("/api/admin/routes")
    def admin_routes(_: dict = Depends(require_supervisor)):
        return {"routes": database.list_routes()}

    @router.post("/api/admin/routes", status_code=201)
    def create_admin_route(
        payload: RouteCreateRequest,
        _: dict = Depends(require_supervisor),
    ):
        code = payload.route_code.strip().upper()
        if any(item["route_code"] == code for item in database.list_routes()):
            raise HTTPException(status_code=409, detail="Mã tuyến đã tồn tại")
        return {"route": database.create_route(
            code, payload.name.strip(), payload.start_location.strip(),
            payload.end_location.strip(), payload.distance_km,
        )}

    @router.patch("/api/admin/routes/{route_id}")
    def update_admin_route(
        route_id: int,
        payload: RouteUpdateRequest,
        _: dict = Depends(require_supervisor),
    ):
        if database.get_route(route_id) is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy tuyến đường")
        code = payload.route_code.strip().upper()
        if any(
            int(item["id"]) != route_id and item["route_code"] == code
            for item in database.list_routes()
        ):
            raise HTTPException(status_code=409, detail="Mã tuyến đã tồn tại")
        route = database.update_route(
            route_id, code, payload.name.strip(), payload.start_location.strip(),
            payload.end_location.strip(), payload.distance_km, payload.active,
        )
        return {"route": route}

    @router.delete("/api/admin/routes/{route_id}")
    def delete_admin_route(
        route_id: int,
        _: dict = Depends(require_supervisor),
    ):
        if database.get_route(route_id) is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy tuyến đường")
        if database.route_trip_count(route_id):
            raise HTTPException(
                status_code=409,
                detail="Không thể xóa tuyến đã được dùng trong chuyến. Hãy ngừng hoạt động tuyến để giữ lịch sử.",
            )
        database.delete_route(route_id)
        return {"deleted": True, "route_id": route_id}

    @router.get("/api/admin/trips")
    def admin_trips(
        trip_status: str | None = Query(
            default=None, alias="status", pattern=r"^(planned|running|completed|cancelled)$"
        ),
        _: dict = Depends(require_supervisor),
    ):
        return {"trips": database.list_trips(trip_status)}

    @router.post("/api/admin/trips", status_code=201)
    def create_admin_trip(
        payload: TripCreateRequest,
        _: dict = Depends(require_supervisor),
    ):
        if database.get_driver(payload.driver_id) is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy tài xế")
        if database.get_vehicle(payload.vehicle_id) is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy phương tiện")
        if payload.route_id is not None and database.get_route(payload.route_id) is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy tuyến đường")
        if any(item["trip_code"] == payload.trip_code.strip().upper() for item in database.list_trips()):
            raise HTTPException(status_code=409, detail="Mã chuyến đã tồn tại")
        return {"trip": database.create_trip(
            payload.trip_code.strip().upper(), payload.driver_id, payload.vehicle_id,
            payload.route_id, payload.planned_start_at,
        )}

    @router.patch("/api/admin/trips/{trip_id}")
    def update_admin_trip(
        trip_id: int,
        payload: TripUpdateRequest,
        _: dict = Depends(require_supervisor),
    ):
        current = database.get_trip(trip_id)
        if current is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy chuyến đi")
        if current["status"] != "planned":
            raise HTTPException(status_code=409, detail="Chỉ được sửa chuyến đang ở trạng thái dự kiến")
        if database.get_driver(payload.driver_id) is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy tài xế")
        if database.get_vehicle(payload.vehicle_id) is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy phương tiện")
        if payload.route_id is not None and database.get_route(payload.route_id) is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy tuyến đường")
        code = payload.trip_code.strip().upper()
        if any(
            int(item["id"]) != trip_id and item["trip_code"] == code
            for item in database.list_trips()
        ):
            raise HTTPException(status_code=409, detail="Mã chuyến đã tồn tại")
        trip = database.update_planned_trip(
            trip_id, code, payload.driver_id, payload.vehicle_id,
            payload.route_id, payload.planned_start_at,
        )
        if trip is None:
            raise HTTPException(status_code=409, detail="Chuyến không còn ở trạng thái dự kiến")
        return {"trip": database.get_trip(trip_id)}

    @router.delete("/api/admin/trips/{trip_id}")
    def delete_admin_trip(
        trip_id: int,
        _: dict = Depends(require_supervisor),
    ):
        trip = database.get_trip(trip_id)
        if trip is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy chuyến đi")
        if trip["status"] != "planned":
            raise HTTPException(status_code=409, detail="Chỉ được xóa chuyến đang ở trạng thái dự kiến")
        if sum(database.trip_dependency_counts(trip_id).values()):
            raise HTTPException(
                status_code=409,
                detail="Không thể xóa chuyến đã có phiên giám sát hoặc dữ liệu GPS",
            )
        database.delete_trip(trip_id)
        return {"deleted": True, "trip_id": trip_id}

    @router.delete("/api/admin/trips/{trip_id}/with-data")
    def delete_admin_finished_trip_with_data(
        trip_id: int,
        _: dict = Depends(require_supervisor),
    ):
        trip = database.get_trip(trip_id)
        if trip is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy chuyến đi")
        if trip["status"] not in {"completed", "cancelled"}:
            raise HTTPException(
                status_code=409,
                detail="Chỉ được xóa toàn bộ dữ liệu của chuyến đã hoàn thành hoặc đã hủy",
            )
        session_ids = database.delete_trip_with_dependencies(trip_id)
        session_root = settings.session_output_root.resolve()
        for session_id in session_ids:
            output_dir = (session_root / str(session_id)).resolve()
            if output_dir.parent == session_root and output_dir.exists():
                shutil.rmtree(output_dir)
        return {
            "deleted": True,
            "trip_id": trip_id,
            "deleted_sessions": len(session_ids),
        }

    @router.patch("/api/admin/trips/{trip_id}/status")
    def update_admin_trip_status(
        trip_id: int, payload: TripStatusRequest,
        _: dict = Depends(require_supervisor),
    ):
        current = database.get_trip(trip_id)
        if current is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy chuyến đi")
        allowed = {
            "planned": {"running", "cancelled"},
            "running": {"completed", "cancelled"},
            "completed": set(), "cancelled": set(),
        }
        if payload.status != current["status"] and payload.status not in allowed[current["status"]]:
            raise HTTPException(status_code=409, detail="Chuyển trạng thái chuyến không hợp lệ")
        if payload.status == "running" and current["status"] != "running":
            driver = database.get_driver(int(current["driver_id"]))
            vehicle = database.get_vehicle(int(current["vehicle_id"]))
            if driver is None or not bool(driver["active"]):
                raise HTTPException(
                    status_code=409,
                    detail="Tài xế của chuyến đang bị khóa hoặc không còn tồn tại",
                )
            if vehicle is None or not bool(vehicle["active"]):
                raise HTTPException(status_code=409, detail="Phương tiện của chuyến đang bị khóa hoặc không còn tồn tại")
            device = database.get_active_edge_device_for_vehicle(int(current["vehicle_id"]))
            if device is None:
                raise HTTPException(status_code=409, detail="Phương tiện chưa có thiết bị Edge đang hoạt động")
            conflict = database.find_running_trip_conflict(
                int(current["driver_id"]), int(current["vehicle_id"]), trip_id,
            )
            if conflict is not None:
                if int(conflict["driver_id"]) == int(current["driver_id"]):
                    detail = f"Tài xế đang thực hiện chuyến {conflict['trip_code']}"
                else:
                    detail = f"Phương tiện đang thực hiện chuyến {conflict['trip_code']}"
                raise HTTPException(status_code=409, detail=detail)
        trip = database.update_trip_status(trip_id, payload.status)
        vehicle = database.get_vehicle(int(current["vehicle_id"]))
        if vehicle is not None:
            vehicle_status = "assigned" if payload.status == "running" else "available"
            database.update_vehicle(
                int(vehicle["id"]), vehicle["vehicle_type"], vehicle.get("model"),
                vehicle_status, bool(vehicle["active"]),
            )
        return {"trip": database.get_trip(int(trip["id"]))}

    @router.get("/api/admin/trips/{trip_id}/locations")
    def admin_trip_locations(
        trip_id: int, limit: int = Query(default=1000, ge=1, le=5000),
        _: dict = Depends(require_supervisor),
    ):
        if database.get_trip(trip_id) is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy chuyến đi")
        return {"locations": database.list_trip_locations(trip_id, limit), "events": database.trip_events(trip_id)}

    @router.post("/api/edge/trips/{trip_id}/locations", status_code=201)
    def add_edge_trip_location(
        trip_id: int, payload: VehicleLocationRequest,
        device: dict = Depends(require_edge_device),
    ):
        trip = database.get_trip(trip_id)
        if trip is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy chuyến đi")
        if trip["status"] != "running":
            raise HTTPException(status_code=409, detail="Chỉ ghi GPS cho chuyến đang chạy")
        if device.get("vehicle_id") is None:
            raise HTTPException(status_code=409, detail="Thiết bị Edge chưa được gắn với phương tiện")
        if int(device["vehicle_id"]) != int(trip["vehicle_id"]):
            raise HTTPException(status_code=409, detail="Thiết bị Edge không thuộc phương tiện của chuyến")
        database.touch_edge_device(int(device["id"]))
        return {"location": database.add_vehicle_location(
            trip_id, payload.latitude, payload.longitude,
            payload.speed_kph, payload.heading, payload.recorded_at, payload.source,
        )}

    @router.get("/api/live/frame.jpg", response_class=Response)
    def live_frame(
        session_id: int | None = Query(default=None, ge=1),
        _: dict = Depends(require_supervisor),
    ):
        preview = manager.latest_preview(session_id)
        if preview is None:
            raise HTTPException(status_code=404, detail="Chưa có frame trực tiếp")
        session_id, jpeg = preview
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "X-DMS-Session-ID": str(session_id),
            },
        )

    @router.post("/api/videos/upload")
    async def upload_test_video(
        file: UploadFile = File(...),
        _: dict = Depends(require_supervisor),
    ):
        original_name = Path(file.filename or "video").name
        suffix = Path(original_name).suffix.lower()
        if suffix not in allowed_video_extensions:
            raise HTTPException(status_code=415, detail="Chỉ hỗ trợ MP4, AVI, MOV hoặc MKV")

        upload_dir = settings.project_root / "data" / "test_videos" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        stored_name = f"{uuid4().hex}{suffix}"
        target = upload_dir / stored_name
        uploaded = 0
        try:
            with target.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    uploaded += len(chunk)
                    if uploaded > max_upload_bytes:
                        raise HTTPException(status_code=413, detail="Video vượt quá giới hạn 500 MB")
                    output.write(chunk)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

        return {
            "source": str(target.relative_to(settings.project_root)).replace("\\", "/"),
            "filename": original_name,
            "size_bytes": uploaded,
        }
    @router.post("/api/sessions/start", response_model=StartSessionResponse, status_code=202)
    def start_session(
        payload: StartSessionRequest,
        _: dict = Depends(require_supervisor),
    ):
        if not settings.allow_backend_inference:
            raise HTTPException(
                status_code=409,
                detail="Luồng chính thức chạy AI tại Edge. Hãy dùng scripts/20_run_edge_simulator.py.",
            )
        trip = database.get_trip(payload.trip_id)
        if trip is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy chuyến đi")
        if trip["status"] != "running":
            raise HTTPException(status_code=409, detail="Chuyến phải ở trạng thái đang chạy")
        try:
            session_id = manager.start(
                payload.source,
                payload.device,
                payload.event_cooldown_seconds,
                payload.save_video,
                payload.yolo_interval,
                payload.cnn_interval,
                int(trip["id"]),
            )
            session = database.get_session(session_id)
            return StartSessionResponse(session_id=session_id, status=session["status"], source=session["source"])
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy video: {error}") from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.post("/api/sessions/{session_id}/stop")
    def stop_session(session_id: int, _: dict = Depends(require_supervisor)):
        session = database.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy phiên")
        if not manager.stop(session_id):
            raise HTTPException(status_code=409, detail="Phiên không chạy hoặc không phải phiên hiện tại")
        return {"session_id": session_id, "stop_requested": True}

    @router.post("/api/sessions/stop")
    def stop_active_session(_: dict = Depends(require_supervisor)):
        session_id = manager.active_session_id
        if session_id is None or not manager.stop(session_id):
            raise HTTPException(status_code=409, detail="Không có phiên đang chạy")
        return {"session_id": session_id, "stop_requested": True}

    @router.get("/api/sessions")
    def sessions(
        limit: int = Query(default=50, ge=1, le=200),
        driver_id: int | None = Query(default=None, ge=1),
        _: dict = Depends(require_supervisor),
    ):
        return {"sessions": database.list_sessions(limit, driver_id)}

    @router.get("/api/metrics/summary")
    def metrics_summary(_: dict = Depends(require_supervisor)):
        return database.metrics_summary()

    @router.get("/api/sessions/{session_id}")
    def session_detail(session_id: int, _: dict = Depends(require_supervisor)):
        session = database.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy phiên")
        return {"session": session}

    @router.delete("/api/sessions/{session_id}")
    def delete_session(session_id: int, _: dict = Depends(require_supervisor)):
        session = database.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy phiên")
        if manager.active_session_id == session_id or session["status"] in {"running", "queued"}:
            raise HTTPException(status_code=409, detail="Không thể xóa phiên đang chạy")
        session_root = settings.session_output_root.resolve()
        output_dir = (session_root / str(session_id)).resolve()
        if output_dir.parent != session_root:
            raise HTTPException(status_code=400, detail="Đường dẫn phiên không hợp lệ")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        database.delete_session(session_id)
        return {"session_id": session_id, "deleted": True}

    @router.get("/api/sessions/{session_id}/events")
    def session_events(session_id: int, _: dict = Depends(require_supervisor)):
        if database.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy phiên")
        return {"events": database.list_events(session_id)}

    @router.get("/api/sessions/{session_id}/metrics")
    def session_metrics(session_id: int, _: dict = Depends(require_supervisor)):
        if database.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy phiên")
        return {
            "summary": database.frame_metrics_summary(session_id),
            "samples": database.list_frame_metrics(session_id),
        }

    @router.get("/api/sessions/{session_id}/report.html", response_class=HTMLResponse)
    def session_html_report(session_id: int, _: dict = Depends(require_supervisor)):
        session = database.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy phiên")
        return HTMLResponse(build_session_report(
            session,
            database.frame_metrics_summary(session_id),
            database.list_frame_metrics(session_id),
            database.list_events(session_id),
        ))

    @router.get("/api/events/{event_id}/snapshot.jpg")
    def event_snapshot(event_id: int, user: dict = Depends(current_user)):
        event = ensure_event_access(database.get_event(event_id), user)
        if event is None or not event.get("snapshot_path"):
            raise HTTPException(status_code=404, detail="Sự kiện không có snapshot")
        path = (settings.project_root / event["snapshot_path"]).resolve()
        if not path.is_relative_to(settings.session_output_root.resolve()) or not path.is_file():
            raise HTTPException(status_code=404, detail="Không tìm thấy snapshot")
        return FileResponse(path, media_type="image/jpeg", filename=path.name)

    @router.get("/api/events/{event_id}/clip.mp4")
    def event_clip(event_id: int, user: dict = Depends(current_user)):
        event = ensure_event_access(database.get_event(event_id), user)
        if event is None or not event.get("clip_path"):
            raise HTTPException(status_code=404, detail="Sự kiện không có clip")
        path = (settings.project_root / event["clip_path"]).resolve()
        if not path.is_relative_to(settings.session_output_root.resolve()) or not path.is_file():
            raise HTTPException(status_code=404, detail="Clip đang được tạo hoặc không tồn tại")
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    @router.patch("/api/events/{event_id}/review")
    def review_detection_event(
        event_id: int,
        payload: EventReviewRequest,
        supervisor: dict = Depends(require_supervisor),
    ):
        event = database.review_event(
            event_id, payload.status, payload.note.strip() if payload.note else None,
            int(supervisor["id"]),
        )
        if event is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy sự kiện")
        manager.publish_event_review(int(event["session_id"]), event)
        return {"event": event}

    @router.get("/api/audio-alerts/{warning}.wav")
    def vietnamese_audio_alert(warning: str, _: dict = Depends(require_supervisor)):
        if warning not in {*ALERT_MESSAGES, "danger"}:
            raise HTTPException(status_code=404, detail="Không tìm thấy âm thanh cảnh báo")
        path = Path(__file__).resolve().parents[1] / "services" / f"audio_vi_{warning}.wav"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Chưa tạo âm thanh cảnh báo")
        return FileResponse(path, media_type="audio/wav", filename=path.name)

    @router.get("/api/sessions/{session_id}/report.csv")
    def session_report(
        session_id: int,
        row_type: str = Query("all", pattern="^(all|event|metric)$"),
        sort_by: str = Query(
            "timestamp_seconds",
            pattern="^(timestamp_seconds|frame_index|risk_score|row_type)$",
        ),
        order: str = Query("asc", pattern="^(asc|desc)$"),
        _: dict = Depends(require_supervisor),
    ):
        session = database.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy phiên")

        summary = database.frame_metrics_summary(session_id)
        session_values = {
            "session_id": session["id"],
            "driver_id": session.get("driver_id"),
            "driver_external_id": session.get("driver_external_id"),
            "driver_name": session.get("driver_name"),
            "vehicle_plate": session.get("plate_number"),
            "trip_code": session.get("trip_code"),
            "source": session.get("source"),
            "session_status": session.get("status"),
            "device": session.get("device"),
            "session_frame_count": session.get("frame_count"),
            "session_avg_fps": session.get("avg_fps"),
            "session_avg_risk_score": session.get("avg_risk_score"),
            "session_max_risk_score": session.get("max_risk_score"),
            "face_detection_rate": session.get("face_detection_rate"),
            "metric_sample_count": summary.get("sample_count"),
        }
        rows: list[dict] = []
        if row_type in {"all", "event"}:
            for event in database.list_events(session_id):
                rows.append({
                    **session_values,
                    "row_type": "event",
                    "event_id": event.get("id"),
                    "occurred_at": event.get("occurred_at"),
                    "frame_index": event.get("frame_index"),
                    "timestamp_seconds": event.get("timestamp_seconds"),
                    "event_type": event.get("event_type"),
                    "confidence": event.get("confidence"),
                    "risk_score": event.get("risk_score"),
                    "severity": event.get("severity"),
                    "review_status": event.get("review_status") or "new",
                    "review_note": event.get("review_note"),
                    "reviewed_at": event.get("reviewed_at"),
                    "snapshot_path": event.get("snapshot_path"),
                    "clip_path": event.get("clip_path"),
                })
        if row_type in {"all", "metric"}:
            for sample in database.list_frame_metrics(session_id):
                rows.append({
                    **session_values,
                    **{key: value for key, value in sample.items() if key != "id"},
                    "row_type": "metric",
                })

        def sort_value(row: dict):
            value = row.get(sort_by)
            if value is None:
                return (1, 0)
            if sort_by == "row_type":
                return (0, str(value))
            try:
                return (0, float(value))
            except (TypeError, ValueError):
                return (0, str(value))

        rows.sort(key=sort_value, reverse=order == "desc")
        columns = [
            "row_type", "session_id", "driver_id", "driver_external_id", "driver_name",
            "vehicle_plate", "trip_code", "source", "session_status", "device",
            "session_frame_count", "session_avg_fps", "session_avg_risk_score",
            "session_max_risk_score", "face_detection_rate", "metric_sample_count",
            "event_id", "occurred_at", "frame_index", "timestamp_seconds",
            "event_type", "confidence", "risk_score", "severity", "review_status",
            "review_note", "reviewed_at", "ear_left", "ear_right", "ear", "mar",
            "geometry_eye_closed", "yawning", "blink_count", "yawn_count", "cnn_awake",
            "cnn_drowsy", "lstm_awake", "lstm_drowsy", "perclos", "phone", "cigarette",
            "seatbelt", "yolo_eye_closed", "yolo_eye_open", "warnings", "processing_fps",
            "pitch", "yaw", "roll", "gaze_x", "gaze_y", "eye_closure_seconds",
            "max_eye_closure_seconds", "microsleep", "head_distracted", "gaze_distracted",
            "distraction_score", "snapshot_path", "clip_path",
        ]
        output = io.StringIO(newline="")
        output.write("sep=;\r\n")
        writer = csv.writer(output, delimiter=";", lineterminator="\r\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row.get(column) for column in columns])
        return Response(
            content="\ufeff" + output.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="dms_session_{session_id}.csv"'},
        )

    @router.websocket("/ws/monitor")
    async def monitor(websocket: WebSocket):
        await websocket.accept()
        try:
            auth_message = await asyncio.wait_for(websocket.receive_json(), timeout=5)
            if not isinstance(auth_message, dict):
                raise ValueError("Thông điệp xác thực không hợp lệ")
            token = auth_message.get("access_token") if auth_message.get("type") == "auth" else None
            if not isinstance(token, str):
                raise ValueError("Thiếu access token")
            user = authenticated_user_from_token(token)
            if user["role"] != "supervisor":
                await websocket.close(code=4403, reason="Cần quyền người giám sát")
                return
        except WebSocketDisconnect:
            return
        except (asyncio.TimeoutError, ValueError, TypeError, KeyError):
            await websocket.close(code=4401, reason="Phiên đăng nhập không hợp lệ")
            return
        channel = manager.subscribe()
        try:
            await websocket.send_json({"type": "connected", "active_session_id": manager.active_session_id})
            while True:
                try:
                    message = await asyncio.to_thread(channel.get, True, 15)
                    await websocket.send_json(message)
                except queue.Empty:
                    await websocket.send_json({"type": "heartbeat", "active_session_id": manager.active_session_id})
        except WebSocketDisconnect:
            pass
        finally:
            manager.unsubscribe(channel)

    return router
