import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dms_inference.features.eye_state import PerclosTracker
from dms_inference.features.face_geometry import eye_aspect_ratio, mouth_aspect_ratio
from dms_inference.scoring.driver_state_engine import DriverStateEngine
from dms_inference.temporal.sequence_buffer import SequenceBuffer
from dms_inference.temporal.temporal_filter import TemporalFilter
from dms_inference.tracking.bbox_smoother import BBoxSmoother


class InferenceLogicTests(unittest.TestCase):
    def test_sequence_buffer_shape_and_readiness(self):
        buffer = SequenceBuffer(sequence_length=2, feature_dim=3)
        buffer.push(np.array([1, 2, 3]))
        self.assertFalse(buffer.ready)
        buffer.push(np.array([4, 5, 6]))
        self.assertEqual(buffer.get().shape, (2, 3))
        with self.assertRaises(ValueError):
            buffer.push(np.array([1, 2]))
        buffer.reset()
        self.assertEqual(len(buffer), 0)

    def test_bbox_smoother(self):
        smoother = BBoxSmoother(alpha=0.5)
        self.assertEqual(smoother.update((0, 0, 10, 10)), (0, 0, 10, 10))
        self.assertEqual(smoother.update((10, 10, 20, 20)), (5, 5, 15, 15))

    def test_temporal_filter_hysteresis(self):
        filt = TemporalFilter(4, enter_drowsy=0.7, exit_drowsy=0.3, min_observations=2)
        self.assertEqual(filt.update(0.9)["state"], "awake")
        self.assertEqual(filt.update(0.9)["state"], "drowsy")
        filt.update(0.0)
        self.assertEqual(filt.update(0.0)["state"], "drowsy")
        filt.update(0.0)
        self.assertEqual(filt.update(0.0)["state"], "awake")

    def test_perclos_uses_only_valid_observations(self):
        tracker = PerclosTracker(window_size=3, min_observations=2)
        tracker.update(None)
        result = tracker.update(True)
        self.assertFalse(result["ready"])
        result = tracker.update(False)
        self.assertTrue(result["ready"])
        self.assertEqual(result["value"], 0.5)

    def test_perclos_uses_camera_time_window(self):
        tracker = PerclosTracker(
            min_observations=3, window_seconds=60, min_duration_seconds=10,
        )
        self.assertFalse(tracker.update(True, 0)["ready"])
        self.assertFalse(tracker.update(False, 5)["ready"])
        result = tracker.update(True, 10)
        self.assertTrue(result["ready"])
        self.assertAlmostEqual(result["value"], 2 / 3)
        result = tracker.update(False, 71)
        self.assertFalse(result["ready"])
        self.assertEqual(result["observations"], 1)

    def test_scoring_does_not_infer_no_seatbelt_by_default(self):
        engine = DriverStateEngine()
        result = engine.evaluate(
            {"ready": True, "state": "drowsy"},
            {"phone": True, "cigarette": False, "seatbelt": False},
            {"ready": False, "value": 0.0},
        )
        self.assertEqual(result["risk_score"], 80)
        self.assertNotIn("no_seatbelt", result["warnings"])

    def test_eye_aspect_ratio_geometry(self):
        points = np.array([[0, 0], [1, 1], [3, 1], [4, 0], [3, -1], [1, -1]], dtype=float)
        self.assertAlmostEqual(eye_aspect_ratio(points), 0.5)

    def test_mouth_aspect_ratio_geometry(self):
        points = np.array([[0, 0], [4, 0], [1, 1], [2, 1], [3, 1], [1, -1], [2, -1], [3, -1]], dtype=float)
        self.assertAlmostEqual(mouth_aspect_ratio(points), 0.5)

    def test_yawn_adds_risk(self):
        result = DriverStateEngine().evaluate(
            {"ready": False, "state": "unknown"},
            {"phone": False, "cigarette": False, "seatbelt": True, "eye_closed": False},
            {"ready": False, "value": 0.0},
            {"yawning": True},
        )
        self.assertIn("yawn", result["warnings"])
        self.assertEqual(result["risk_score"], 15)

    def test_open_eye_distraction_adds_risk(self):
        result = DriverStateEngine().evaluate(
            {"ready": False, "state": "unknown"},
            {"phone": False, "cigarette": False, "seatbelt": True, "eye_closed": False},
            {"ready": False, "value": 0.0},
            {"microsleep": False, "eye_closed": False, "yawning": False, "head_distracted": True, "gaze_distracted": False},
        )
        self.assertEqual(result["warnings"], ["distraction"])
        self.assertEqual(result["risk_score"], 20)

    def test_yawn_and_closed_eyes_do_not_duplicate_distraction(self):
        engine = DriverStateEngine()
        yawning = engine.evaluate(
            {"ready": False, "state": "unknown"},
            {"phone": False, "cigarette": False, "seatbelt": True, "eye_closed": False},
            {"ready": False, "value": 0.0},
            {"yawning": True, "eye_closed": False, "head_distracted": True, "gaze_distracted": True},
        )
        self.assertEqual(yawning["warnings"], ["yawn"])
        closed = engine.evaluate(
            {"ready": False, "state": "unknown"},
            {"phone": False, "cigarette": False, "seatbelt": True, "eye_closed": True},
            {"ready": False, "value": 0.0},
            {"yawning": False, "eye_closed": True, "head_distracted": True, "gaze_distracted": True},
        )
        self.assertNotIn("distraction", closed["warnings"])

    def test_specific_fatigue_warning_prevents_duplicate_eye_closure_events(self):
        result = DriverStateEngine().evaluate(
            {"ready": True, "state": "drowsy"},
            {"phone": False, "cigarette": False, "seatbelt": True, "eye_closed": True},
            {"ready": True, "value": 0.8},
            {"microsleep": True, "eye_closed": True, "yawning": False},
        )
        self.assertEqual(result["warnings"], ["microsleep"])
        self.assertEqual(result["risk_score"], 50)

    def test_phone_prevents_duplicate_generic_distraction_event(self):
        result = DriverStateEngine().evaluate(
            {"ready": False, "state": "unknown"},
            {"phone": True, "cigarette": False, "seatbelt": True, "eye_closed": False},
            {"ready": False, "value": 0.0},
            {"microsleep": False, "eye_closed": False, "yawning": False, "head_distracted": True},
        )
        self.assertEqual(result["warnings"], ["phone"])
        self.assertEqual(result["risk_score"], 30)


if __name__ == "__main__":
    unittest.main()
