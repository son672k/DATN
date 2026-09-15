from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


def _utc_iso(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Thời gian phải theo định dạng ISO 8601") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat()


class StartSessionRequest(BaseModel):
    source: str = Field(min_length=1)
    trip_id: int = Field(ge=1)
    device: str | None = None
    event_cooldown_seconds: float = Field(default=3.0, ge=0.0, le=60.0)
    save_video: bool = False
    yolo_interval: int = Field(default=2, ge=1, le=10)
    cnn_interval: int = Field(default=2, ge=1, le=10)


class StartSessionResponse(BaseModel):
    session_id: int
    status: str
    source: str


class AdminDriverAccountRequest(BaseModel):
    """Driver profile plus the credentials issued by a supervisor."""

    external_id: str = Field(min_length=2, max_length=80)
    display_name: str = Field(min_length=1, max_length=120)
    phone: str | None = Field(default=None, max_length=32)
    username: str = Field(min_length=3, max_length=80, pattern=r"^[A-Za-z0-9._-]+$")
    password: str = Field(min_length=8, max_length=128)


class UserActiveRequest(BaseModel):
    active: bool


class EdgeDeviceCreateRequest(BaseModel):
    device_uid: str = Field(min_length=3, max_length=120, pattern=r"^[A-Za-z0-9._-]+$")
    label: str = Field(min_length=2, max_length=120)
    vehicle_id: int | None = Field(default=None, ge=1)


class EdgeHeartbeatRequest(BaseModel):
    software_version: str | None = Field(default=None, max_length=40)


class EdgeSessionCreateRequest(BaseModel):
    trip_id: int = Field(ge=1)
    source: str = Field(min_length=1, max_length=500)
    device: str = Field(default="cpu", min_length=1, max_length=80)
    yolo_interval: int = Field(default=2, ge=1, le=10)
    cnn_interval: int = Field(default=2, ge=1, le=10)
    save_video: bool = False
    replace_active: bool = False


class EdgeMetricBatchRequest(BaseModel):
    samples: list[dict] = Field(min_length=1, max_length=250)
    live_message: dict | None = None


class EdgeEventCreateRequest(BaseModel):
    client_event_id: str = Field(min_length=8, max_length=160)
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    event_type: str = Field(min_length=2, max_length=100)
    confidence: float = Field(ge=0, le=1)
    risk_score: int = Field(ge=0, le=100)
    payload: dict | None = None


class EdgeSessionCompleteRequest(BaseModel):
    status: str = Field(pattern=r"^(completed|stopped|failed)$")
    frame_count: int = Field(default=0, ge=0)
    video_duration_seconds: float = Field(default=0, ge=0)
    processing_seconds: float = Field(default=0, ge=0)
    avg_fps: float = Field(default=0, ge=0)
    avg_risk_score: float = Field(default=0, ge=0, le=100)
    max_risk_score: int = Field(default=0, ge=0, le=100)
    face_detection_rate: float = Field(default=0, ge=0, le=1)
    error_message: str | None = Field(default=None, max_length=1000)


class EdgeDeviceUpdateRequest(BaseModel):
    label: str = Field(min_length=1, max_length=120)
    vehicle_id: int | None = Field(default=None, ge=1)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


class PasswordResetRequest(BaseModel):
    new_password: str = Field(min_length=8, max_length=128)


class DriverProfileUpdateRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=120)
    phone: str | None = Field(default=None, max_length=32)


class EventReviewRequest(BaseModel):
    status: str = Field(pattern=r"^(new|acknowledged|resolved)$")
    note: str | None = Field(default=None, max_length=500)


class VehicleCreateRequest(BaseModel):
    plate_number: str = Field(min_length=5, max_length=30)
    vehicle_type: str = Field(min_length=2, max_length=80)
    model: str | None = Field(default=None, max_length=120)


class VehicleUpdateRequest(BaseModel):
    plate_number: str = Field(min_length=5, max_length=30)
    vehicle_type: str = Field(min_length=2, max_length=80)
    model: str | None = Field(default=None, max_length=120)
    status: str = Field(pattern=r"^(available|assigned|maintenance|offline)$")
    active: bool = True


class RouteCreateRequest(BaseModel):
    route_code: str = Field(min_length=2, max_length=80)
    name: str = Field(min_length=2, max_length=255)
    start_location: str = Field(min_length=2, max_length=255)
    end_location: str = Field(min_length=2, max_length=255)
    distance_km: float | None = Field(default=None, ge=0, le=10000)


class RouteUpdateRequest(RouteCreateRequest):
    active: bool = True


class TripCreateRequest(BaseModel):
    trip_code: str = Field(min_length=2, max_length=100)
    driver_id: int = Field(ge=1)
    vehicle_id: int = Field(ge=1)
    route_id: int | None = Field(default=None, ge=1)
    planned_start_at: str | None = Field(default=None, max_length=64)

    @field_validator("planned_start_at")
    @classmethod
    def validate_planned_start(cls, value: str | None) -> str | None:
        return _utc_iso(value)


class TripUpdateRequest(TripCreateRequest):
    pass


class TripStatusRequest(BaseModel):
    status: str = Field(pattern=r"^(planned|running|completed|cancelled)$")


class VehicleLocationRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    speed_kph: float | None = Field(default=None, ge=0, le=400)
    heading: float | None = Field(default=None, ge=0, lt=360)
    recorded_at: str | None = Field(default=None, max_length=64)
    source: str = Field(default="edge-simulator", min_length=2, max_length=80)

    @field_validator("recorded_at")
    @classmethod
    def validate_recorded_at(cls, value: str | None) -> str | None:
        return _utc_iso(value)
