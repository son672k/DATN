from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    # MySQL is the only runtime database. Tests instantiate temporary SQLite
    # databases directly and never use this application setting.
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DMS_DATABASE_URL",
            "mysql://dms_app:change-me-before-deploying@127.0.0.1:3306/dms",
        )
    )
    yolo_checkpoint: Path = PROJECT_ROOT / "outputs" / "final_models" / "best_yolo.pt"
    cnn_checkpoint: Path = PROJECT_ROOT / "outputs" / "final_models" / "best_cnn.pth"
    lstm_checkpoint: Path = PROJECT_ROOT / "outputs" / "final_models" / "best_lstm.pth"
    inference_config: Path = PROJECT_ROOT / "configs" / "inference.json"
    face_landmarker: Path = PROJECT_ROOT / "models" / "face_landmarker.task"
    session_output_root: Path = PROJECT_ROOT / "outputs" / "backend_sessions"
    environment: str = field(default_factory=lambda: os.getenv("DMS_ENV", "development"))
    jwt_secret: str = field(
        default_factory=lambda: os.getenv(
            "DMS_JWT_SECRET",
            "development-only-change-this-before-aws-deployment",
        )
    )
    jwt_expire_minutes: int = field(
        default_factory=lambda: int(os.getenv("DMS_JWT_EXPIRE_MINUTES", "720"))
    )
    edge_audio_alert: bool = field(
        default_factory=lambda: os.getenv("DMS_EDGE_AUDIO_ALERT", "true").lower()
        not in {"0", "false", "no", "off"}
    )
    allow_backend_inference: bool = field(
        default_factory=lambda: os.getenv("DMS_ALLOW_BACKEND_INFERENCE", "false").lower()
        in {"1", "true", "yes", "on"}
    )


settings = Settings()
