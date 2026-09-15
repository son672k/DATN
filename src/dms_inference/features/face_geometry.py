from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "face_landmarker.task"


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(points: np.ndarray) -> float:
    """EAR = (vertical 1 + vertical 2) / (2 * horizontal)."""
    horizontal = _distance(points[0], points[3])
    if horizontal <= 1e-6:
        return 0.0
    return (_distance(points[1], points[5]) + _distance(points[2], points[4])) / (2.0 * horizontal)


def mouth_aspect_ratio(points: np.ndarray) -> float:
    """MAR using three inner-lip vertical pairs normalized by mouth width."""
    horizontal = _distance(points[0], points[1])
    if horizontal <= 1e-6:
        return 0.0
    vertical = sum(_distance(points[index], points[index + 3]) for index in range(2, 5))
    return vertical / (3.0 * horizontal)


class FaceGeometryAnalyzer:
    RIGHT_EYE = (33, 160, 158, 133, 153, 144)
    LEFT_EYE = (362, 385, 387, 263, 373, 380)
    # corners, then upper inner lip (left/center/right), then corresponding lower lip
    MOUTH = (78, 308, 82, 13, 312, 87, 14, 317)
    HEAD_POSE = (1, 152, 33, 263, 61, 291)
    RIGHT_IRIS = (468, 469, 470, 471, 472)
    LEFT_IRIS = (473, 474, 475, 476, 477)

    def __init__(
        self,
        model_path: str | Path | None = None,
        ear_threshold: float = 0.20,
        mar_threshold: float = 0.35,
        blink_min_samples: int = 2,
        yawn_min_samples: int = 3,
        yawn_min_seconds: float = 0.8,
        microsleep_seconds: float = 1.5,
        head_pitch_threshold: float = 25.0,
        head_yaw_threshold: float = 30.0,
        gaze_threshold: float = 0.45,
        distraction_min_seconds: float = 3.0,
    ):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError("Thiếu MediaPipe Tasks. Cài mediapipe==0.10.31.") from exc

        self._mp = mp
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Thiếu model Face Landmarker: {self.model_path}")
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self.ear_threshold = float(ear_threshold)
        self.mar_threshold = float(mar_threshold)
        self.blink_min_samples = int(blink_min_samples)
        self.yawn_min_samples = int(yawn_min_samples)
        self.yawn_min_seconds = float(yawn_min_seconds)
        self.microsleep_seconds = float(microsleep_seconds)
        self.head_pitch_threshold = float(head_pitch_threshold)
        self.head_yaw_threshold = float(head_yaw_threshold)
        self.gaze_threshold = float(gaze_threshold)
        self.distraction_min_seconds = float(distraction_min_seconds)
        self.reset()

    def reset(self) -> None:
        self._closed_streak = 0
        self._yawn_streak = 0
        self._yawning = False
        self._yawn_since: float | None = None
        self.blink_count = 0
        self.yawn_count = 0
        self._eye_closed_since: float | None = None
        self._last_timestamp: float | None = None
        self.max_eye_closure_seconds = 0.0
        self._distraction_since: float | None = None

    @staticmethod
    def _points(landmarks, indices: tuple[int, ...], width: int, height: int) -> np.ndarray:
        return np.asarray([(landmarks[index].x * width, landmarks[index].y * height) for index in indices], dtype=np.float32)

    @staticmethod
    def _head_pose(landmarks, width: int, height: int) -> tuple[float, float, float]:
        image_points = FaceGeometryAnalyzer._points(landmarks, FaceGeometryAnalyzer.HEAD_POSE, width, height).astype(np.float64)
        model_points = np.asarray([
            (0.0, 0.0, 0.0), (0.0, -63.6, -12.5), (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1),
        ], dtype=np.float64)
        focal = float(width)
        camera_matrix = np.asarray([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
        success, rotation, _ = cv2.solvePnP(model_points, image_points, camera_matrix, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return 0.0, 0.0, 0.0
        matrix, _ = cv2.Rodrigues(rotation)
        angles = cv2.RQDecomp3x3(matrix)[0]
        pitch, yaw, roll = map(float, angles)
        # solvePnP may return the equivalent flipped pose around the X axis.
        # Fold it into the human-readable [-90, 90] pitch interval.
        if pitch < -90:
            pitch += 180
        elif pitch > 90:
            pitch -= 180
        return pitch, yaw, roll

    @staticmethod
    def _gaze(landmarks, width: int, height: int) -> tuple[float, float]:
        def eye_value(corners: tuple[int, int], upper: int, lower: int, iris_indices: tuple[int, ...]):
            corner_points = FaceGeometryAnalyzer._points(landmarks, corners, width, height)
            iris = FaceGeometryAnalyzer._points(landmarks, iris_indices, width, height).mean(axis=0)
            left, right = float(corner_points[:, 0].min()), float(corner_points[:, 0].max())
            upper_y = landmarks[upper].y * height
            lower_y = landmarks[lower].y * height
            x = ((float(iris[0]) - left) / max(right - left, 1e-6) - 0.5) * 2.0
            y = ((float(iris[1]) - min(upper_y, lower_y)) / max(abs(lower_y - upper_y), 1e-6) - 0.5) * 2.0
            return x, y

        right = eye_value((33, 133), 159, 145, FaceGeometryAnalyzer.RIGHT_IRIS)
        left = eye_value((362, 263), 386, 374, FaceGeometryAnalyzer.LEFT_IRIS)
        return float(np.clip((right[0] + left[0]) / 2, -1, 1)), float(np.clip((right[1] + left[1]) / 2, -1, 1))

    def analyze(self, frame_bgr: np.ndarray, timestamp_seconds: float | None = None) -> dict | None:
        if timestamp_seconds is None:
            timestamp_seconds = 0.0 if self._last_timestamp is None else self._last_timestamp + 1 / 30
        timestamp_seconds = float(timestamp_seconds)
        self._last_timestamp = timestamp_seconds
        height, width = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        result = self._landmarker.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        if not result.face_landmarks:
            self._closed_streak = 0
            self._yawn_streak = 0
            self._yawning = False
            self._yawn_since = None
            self._eye_closed_since = None
            self._distraction_since = None
            return None

        landmarks = result.face_landmarks[0]
        ear_left = eye_aspect_ratio(self._points(landmarks, self.LEFT_EYE, width, height))
        ear_right = eye_aspect_ratio(self._points(landmarks, self.RIGHT_EYE, width, height))
        ear = (ear_left + ear_right) / 2.0

        mouth = self._points(landmarks, self.MOUTH, width, height)
        mar_points = np.vstack((mouth[:2], mouth[2:5], mouth[5:8]))
        mar = mouth_aspect_ratio(mar_points)
        pitch, yaw, roll = self._head_pose(landmarks, width, height)
        gaze_x, gaze_y = self._gaze(landmarks, width, height)

        eye_closed = ear < self.ear_threshold
        if eye_closed:
            self._closed_streak += 1
            if self._eye_closed_since is None:
                self._eye_closed_since = timestamp_seconds
        else:
            if self._closed_streak >= self.blink_min_samples:
                self.blink_count += 1
            self._closed_streak = 0
            self._eye_closed_since = None

        closure_seconds = max(0.0, timestamp_seconds - self._eye_closed_since) if self._eye_closed_since is not None else 0.0
        self.max_eye_closure_seconds = max(self.max_eye_closure_seconds, closure_seconds)
        microsleep = closure_seconds >= self.microsleep_seconds

        if mar >= self.mar_threshold:
            self._yawn_streak += 1
            if self._yawn_since is None:
                self._yawn_since = timestamp_seconds
            if (
                self._yawn_streak >= self.yawn_min_samples
                and timestamp_seconds - self._yawn_since >= self.yawn_min_seconds
                and not self._yawning
            ):
                self._yawning = True
                self.yawn_count += 1
        else:
            self._yawn_streak = 0
            self._yawning = False
            self._yawn_since = None

        head_candidate = abs(pitch) >= self.head_pitch_threshold or abs(yaw) >= self.head_yaw_threshold
        gaze_candidate = abs(gaze_x) >= self.gaze_threshold or abs(gaze_y) >= self.gaze_threshold
        attention_observable = not eye_closed and not self._yawning
        distraction_candidate = attention_observable and (head_candidate or gaze_candidate)
        if distraction_candidate:
            if self._distraction_since is None:
                self._distraction_since = timestamp_seconds
        else:
            self._distraction_since = None
        distraction_confirmed = bool(
            distraction_candidate
            and self._distraction_since is not None
            and timestamp_seconds - self._distraction_since >= self.distraction_min_seconds
        )
        head_distracted = distraction_confirmed and head_candidate
        gaze_distracted = distraction_confirmed and gaze_candidate
        raw_distraction_score = min(100.0, abs(pitch) / self.head_pitch_threshold * 30 + abs(yaw) / self.head_yaw_threshold * 35 + abs(gaze_x) / self.gaze_threshold * 20 + abs(gaze_y) / self.gaze_threshold * 15)
        distraction_score = raw_distraction_score if distraction_confirmed else 0.0

        return {
            "ear_left": round(ear_left, 6),
            "ear_right": round(ear_right, 6),
            "ear": round(ear, 6),
            "mar": round(mar, 6),
            "eye_closed": eye_closed,
            "yawning": self._yawning,
            "blink_count": self.blink_count,
            "yawn_count": self.yawn_count,
            "ear_threshold": self.ear_threshold,
            "mar_threshold": self.mar_threshold,
            "landmark_count": len(landmarks),
            "eye_closure_seconds": round(closure_seconds, 3),
            "max_eye_closure_seconds": round(self.max_eye_closure_seconds, 3),
            "microsleep": microsleep,
            "pitch": round(pitch, 3),
            "yaw": round(yaw, 3),
            "roll": round(roll, 3),
            "gaze_x": round(gaze_x, 4),
            "gaze_y": round(gaze_y, 4),
            "head_distracted": head_distracted,
            "gaze_distracted": gaze_distracted,
            "distraction_score": round(distraction_score, 2),
        }

    def close(self) -> None:
        self._landmarker.close()
