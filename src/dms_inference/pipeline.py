from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from dms_inference.detection.behavior_yolo import BehaviorYoloDetector
from dms_inference.face.mediapipe_face_detector import FaceDetection, MediaPipeFaceDetector
from dms_inference.features.efficientnet_extractor import EfficientNetExtractor
from dms_inference.features.eye_state import PerclosTracker
from dms_inference.features.face_geometry import FaceGeometryAnalyzer
from dms_inference.scoring.driver_state_engine import DriverStateEngine
from dms_inference.temporal.lstm_predictor import LSTMPredictor
from dms_inference.temporal.sequence_buffer import SequenceBuffer
from dms_inference.temporal.temporal_filter import TemporalFilter
from dms_inference.tracking.bbox_smoother import BBoxSmoother


class DriverMonitoringPipeline:
    """One-frame API joining YOLO, CNN and LSTM without mixing responsibilities."""

    def __init__(
        self,
        yolo_checkpoint: Path,
        cnn_checkpoint: Path,
        lstm_checkpoint: Path,
        config_path: Path,
        device: str | None = None,
    ):
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        face_cfg = config["face"]
        yolo_cfg = config["behavior_yolo"]
        temporal_cfg = config["temporal_filter"]
        perclos_cfg = config["perclos"]
        scoring_cfg = config["scoring"]

        self.face_detector = MediaPipeFaceDetector(
            min_confidence=face_cfg["min_confidence"], padding=face_cfg["padding"]
        )
        self.face_geometry = FaceGeometryAnalyzer(**config.get("face_geometry", {}))
        self.face_smoother = BBoxSmoother(face_cfg["bbox_ema_alpha"])
        self.behavior_detector = BehaviorYoloDetector(
            yolo_checkpoint,
            confidence=yolo_cfg["confidence"],
            image_size=yolo_cfg["image_size"],
            phone_confidence=yolo_cfg.get("phone_confidence", 0.55),
            phone_window_size=yolo_cfg.get("phone_window_size", 3),
            phone_min_hits=yolo_cfg.get("phone_min_hits", 2),
            phone_clear_misses=yolo_cfg.get("phone_clear_misses", 2),
            cigarette_confidence=yolo_cfg.get("cigarette_confidence", 0.5),
            cigarette_window_size=yolo_cfg.get("cigarette_window_size", 3),
            cigarette_min_hits=yolo_cfg.get("cigarette_min_hits", 2),
            cigarette_clear_misses=yolo_cfg.get("cigarette_clear_misses", 2),
        )
        self.cnn = EfficientNetExtractor(cnn_checkpoint, self.device)
        self.lstm = LSTMPredictor(lstm_checkpoint, self.device)
        if self.cnn.feature_dim != self.lstm.feature_dim:
            raise ValueError("CNN feature_dim không khớp LSTM input_dim")
        self.sequence = SequenceBuffer(self.lstm.sequence_length, self.lstm.feature_dim)
        self.temporal_filter = TemporalFilter(**temporal_cfg)
        self.perclos = PerclosTracker(**{
            key: value for key, value in perclos_cfg.items() if key != "threshold"
        })
        self.state_engine = DriverStateEngine(
            perclos_threshold=perclos_cfg["threshold"], **scoring_cfg
        )
        self._reset_cache()

    def _reset_cache(self) -> None:
        self._last_behavior = None
        self._last_face = None
        self._last_cnn = None
        self._last_feature = None
        self._last_lstm = None
        self._last_geometry = None
        self._last_temporal = {
            "state": "unknown",
            "ready": False,
            "observations": 0,
        }
        self._has_visual_state = False

    @staticmethod
    def _eye_observation(behavior: dict) -> bool | None:
        if behavior["eye_closed"] and not behavior["eye_open"]:
            return True
        if behavior["eye_open"] and not behavior["eye_closed"]:
            return False
        return None

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        run_behavior: bool = True,
        run_face_cnn: bool = True,
        timestamp_seconds: float | None = None,
    ) -> dict:
        """Process one frame, optionally reusing recent expensive detections.

        Default arguments preserve full-quality offline inference. Realtime
        callers may sample YOLO and CNN independently while still receiving a
        result for every camera frame.
        """
        if run_behavior or self._last_behavior is None:
            self._last_behavior = self.behavior_detector.detect(frame_bgr)
        behavior = self._last_behavior
        # Repeating the last valid eye observation keeps PERCLOS tied to camera
        # time instead of the lower YOLO sampling rate.
        perclos = self.perclos.update(self._eye_observation(behavior), timestamp_seconds)

        if run_face_cnn or not self._has_visual_state:
            self._last_geometry = self.face_geometry.analyze(frame_bgr, timestamp_seconds)
            detected = self.face_detector.detect(frame_bgr)
            self._has_visual_state = True
            if detected is None:
                self.face_smoother.reset()
                self._last_face = None
                self._last_cnn = None
                self._last_feature = None
                self._last_lstm = None
                self._last_temporal = {
                    "state": "unknown",
                    "ready": False,
                    "observations": len(self.sequence),
                }
            else:
                smooth_box = self.face_smoother.update(detected.box)
                face = self.face_detector.crop(
                    frame_bgr, FaceDetection(smooth_box, detected.confidence)
                )
                if face is None:
                    raise RuntimeError("Face crop rỗng sau khi làm mượt bounding box")
                cnn_result = self.cnn.predict_and_extract(face)
                feature = cnn_result.pop("feature")
                self.sequence.push(feature)
                self._last_feature = feature

                lstm_result = None
                temporal = {
                    "state": "unknown",
                    "ready": False,
                    "observations": len(self.sequence),
                }
                if self.sequence.ready:
                    lstm_result = self.lstm.predict(self.sequence.get())
                    probability = lstm_result["probabilities"].get("drowsy", 0.0)
                    temporal = self.temporal_filter.update(probability)
                self._last_face = {
                    "box": smooth_box,
                    "confidence": detected.confidence,
                }
                self._last_cnn = cnn_result
                self._last_lstm = lstm_result
                self._last_temporal = temporal
        elif self._last_feature is not None:
            # CNN is the expensive part. Reusing its latest feature keeps the
            # LSTM and temporal hysteresis aligned with camera time.
            self.sequence.push(self._last_feature)
            if self.sequence.ready:
                self._last_lstm = self.lstm.predict(self.sequence.get())
                probability = self._last_lstm["probabilities"].get("drowsy", 0.0)
                self._last_temporal = self.temporal_filter.update(probability)

        return {
            "face": self._last_face,
            "cnn": self._last_cnn,
            "lstm": self._last_lstm,
            "temporal": self._last_temporal,
            "behavior": behavior,
            "perclos": perclos,
            "geometry": self._last_geometry,
            "decision": self.state_engine.evaluate(
                self._last_temporal, behavior, perclos, self._last_geometry
            ),
        }

    def reset(self) -> None:
        self.face_smoother.reset()
        self.sequence.reset()
        self.temporal_filter.reset()
        self.perclos.reset()
        self.face_geometry.reset()
        self._reset_cache()

    def close(self) -> None:
        self.face_detector.close()
        self.face_geometry.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
