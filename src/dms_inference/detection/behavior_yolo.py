from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ObjectDetection:
    class_id: int
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]


class DetectionPersistenceFilter:
    """Confirm noisy detections over a short window and clear with hysteresis."""

    def __init__(self, window_size: int = 3, min_hits: int = 2, clear_misses: int = 2):
        if window_size < 1 or not 1 <= min_hits <= window_size or clear_misses < 1:
            raise ValueError("Cấu hình persistence filter không hợp lệ")
        self.history: deque[bool] = deque(maxlen=window_size)
        self.min_hits = int(min_hits)
        self.clear_misses = int(clear_misses)
        self.misses = 0
        self.active = False

    def update(self, detected: bool) -> bool:
        detected = bool(detected)
        self.history.append(detected)
        self.misses = 0 if detected else self.misses + 1
        if not self.active and sum(self.history) >= self.min_hits:
            self.active = True
        elif self.active and self.misses >= self.clear_misses:
            self.active = False
            self.history.clear()
        return self.active


class BehaviorYoloDetector:
    REQUIRED_CLASSES = {"closed_eye", "open_eye", "cigarette", "phone", "seatbelt"}

    def __init__(
        self,
        checkpoint: Path,
        confidence: float = 0.25,
        image_size: int = 640,
        phone_confidence: float = 0.55,
        phone_window_size: int = 3,
        phone_min_hits: int = 2,
        phone_clear_misses: int = 2,
        cigarette_confidence: float = 0.5,
        cigarette_window_size: int = 3,
        cigarette_min_hits: int = 2,
        cigarette_clear_misses: int = 2,
    ):
        from ultralytics import YOLO

        self.model = YOLO(str(checkpoint))
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.phone_confidence = float(phone_confidence)
        self.phone_filter = DetectionPersistenceFilter(
            phone_window_size, phone_min_hits, phone_clear_misses
        )
        self.cigarette_confidence = float(cigarette_confidence)
        self.cigarette_filter = DetectionPersistenceFilter(
            cigarette_window_size, cigarette_min_hits, cigarette_clear_misses
        )
        names = self.model.names
        items = names.items() if isinstance(names, dict) else enumerate(names)
        self.names = {
            int(index): self._normalize_name(name) for index, name in items
        }
        missing = self.REQUIRED_CLASSES - set(self.names.values())
        if missing:
            raise RuntimeError(f"YOLO thiếu lớp {sorted(missing)}; hiện có {self.names}")

    @staticmethod
    def _normalize_name(name: str) -> str:
        return str(name).strip().lower().replace(" ", "_").replace("-", "_")

    def detect(self, frame_bgr: np.ndarray) -> dict:
        result = self.model.predict(
            source=frame_bgr,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )[0]
        detections = []
        max_confidence = {name: 0.0 for name in self.REQUIRED_CLASSES}
        for box in result.boxes:
            class_id = int(box.cls.item())
            name = self.names[class_id]
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].cpu().tolist()]
            detection = ObjectDetection(class_id, name, confidence, (x1, y1, x2, y2))
            detections.append(detection)
            if name in max_confidence:
                max_confidence[name] = max(max_confidence[name], confidence)
        raw_phone = max_confidence["phone"] >= self.phone_confidence
        raw_cigarette = max_confidence["cigarette"] >= self.cigarette_confidence
        return {
            "detections": [asdict(item) for item in detections],
            "phone": self.phone_filter.update(raw_phone),
            "phone_raw": raw_phone,
            "cigarette": self.cigarette_filter.update(raw_cigarette),
            "seatbelt": max_confidence["seatbelt"] > 0,
            "eye_closed": max_confidence["closed_eye"] > 0,
            "eye_open": max_confidence["open_eye"] > 0,
            "confidence": max_confidence,
        }
