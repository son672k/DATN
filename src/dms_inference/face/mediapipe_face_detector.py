from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "models"
    / "blaze_face_short_range.tflite"
)


@dataclass(frozen=True)
class FaceDetection:
    box: tuple[int, int, int, int]
    confidence: float


class MediaPipeFaceDetector:
    """MediaPipe Tasks face ROI detector; not a model trained by this project."""

    def __init__(
        self,
        min_confidence: float = 0.5,
        model_selection: int = 0,
        padding: float = 0.08,
        model_path: str | Path | None = None,
    ):
        del model_selection  # Kept for backward-compatible callers.
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError(
                "Thiếu MediaPipe Tasks. Cài mediapipe==0.10.31."
            ) from exc

        self._mp = mp
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Thiếu model MediaPipe Face Detector: {self.model_path}. "
                "Hãy tải blaze_face_short_range.tflite vào thư mục models/."
            )
        options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=float(min_confidence),
        )
        self._detector = vision.FaceDetector.create_from_options(options)
        self.padding = float(padding)

    @staticmethod
    def _clamp_box(box, width: int, height: int):
        x1, y1, x2, y2 = box
        return (
            max(0, min(width - 1, int(x1))),
            max(0, min(height - 1, int(y1))),
            max(1, min(width, int(x2))),
            max(1, min(height, int(y2))),
        )

    def detect(self, frame_bgr: np.ndarray) -> FaceDetection | None:
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        height, width = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb,
        )
        result = self._detector.detect(mp_image)
        candidates = []
        for detection in result.detections:
            raw = detection.bounding_box
            x1, y1 = raw.origin_x, raw.origin_y
            x2, y2 = x1 + raw.width, y1 + raw.height
            pad_x = (x2 - x1) * self.padding
            pad_y = (y2 - y1) * self.padding
            box = self._clamp_box(
                (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y),
                width,
                height,
            )
            confidence = (
                float(detection.categories[0].score)
                if detection.categories
                else 0.0
            )
            area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
            candidates.append((area, confidence, box))
        if not candidates:
            return None
        _, confidence, box = max(candidates)
        return FaceDetection(box=box, confidence=confidence)

    @staticmethod
    def crop(frame_bgr: np.ndarray, detection: FaceDetection) -> np.ndarray | None:
        x1, y1, x2, y2 = detection.box
        crop = frame_bgr[y1:y2, x1:x2]
        return crop if crop.size else None

    def close(self) -> None:
        self._detector.close()
