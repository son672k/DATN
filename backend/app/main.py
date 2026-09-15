from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import create_router
from .config import settings
from .database import Database
from .services.inference_service import InferenceManager


database = Database(settings.database_url)
manager = InferenceManager(database)


async def edge_presence_monitor() -> None:
    while True:
        await asyncio.sleep(15)
        await asyncio.to_thread(database.mark_stale_edge_devices_offline, 60)


async def daily_report_monitor() -> None:
    from datetime import datetime, timedelta
    import logging
    from .services.safety_service import LOCAL
    from .services.pdf_report_service import daily_pdf, save_atomic
    while True:
        try:
            day = (datetime.now(LOCAL).date() - timedelta(days=1)).isoformat()
            target = settings.project_root / "outputs" / "daily_reports" / f"dms_{day}.pdf"
            if not target.exists():
                content = await asyncio.to_thread(daily_pdf, database, day, settings.project_root)
                await asyncio.to_thread(save_atomic, target, content)
        except Exception:
            logging.getLogger(__name__).exception("Không tạo được báo cáo ngày; sẽ thử lại")
        await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.environment.lower() == "production" and (
        settings.jwt_secret.startswith("development-only")
        or len(settings.jwt_secret) < 32
    ):
        raise RuntimeError("Production yêu cầu DMS_JWT_SECRET ngẫu nhiên, tối thiểu 32 ký tự")
    database.initialize()
    database.recover_interrupted_local_sessions()
    for path in (
        settings.yolo_checkpoint,
        settings.cnn_checkpoint,
        settings.lstm_checkpoint,
        settings.inference_config,
        settings.face_landmarker,
    ):
        if not path.is_file():
            raise RuntimeError(f"Thiếu file bắt buộc: {path}")
    presence_task = asyncio.create_task(edge_presence_monitor())
    report_task = asyncio.create_task(daily_report_monitor())
    try:
        yield
    finally:
        report_task.cancel()
        try:
            await report_task
        except asyncio.CancelledError:
            pass
        presence_task.cancel()
        try:
            await presence_task
        except asyncio.CancelledError:
            pass
        manager.shutdown()


app = FastAPI(
    title="DMS Backend",
    version="1.0.0",
    description="Backend giám sát tài xế sử dụng YOLO, CNN và LSTM.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3002",
    ],
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):[0-9]{4,5}$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(create_router(database, manager))


@app.get("/")
def root():
    return {"service": "DMS Backend", "docs": "/docs", "health": "/api/health"}
