from __future__ import annotations

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dms_inference.detection.behavior_yolo import DetectionPersistenceFilter


class DetectionPersistenceFilterTests(unittest.TestCase):
    def test_phone_requires_two_hits_and_two_misses_to_clear(self):
        phone = DetectionPersistenceFilter(window_size=3, min_hits=2, clear_misses=2)

        self.assertFalse(phone.update(True))
        self.assertFalse(phone.update(False))
        self.assertTrue(phone.update(True))
        self.assertTrue(phone.update(False))
        self.assertFalse(phone.update(False))

    def test_filter_can_activate_again_after_clear(self):
        phone = DetectionPersistenceFilter(window_size=3, min_hits=2, clear_misses=2)
        for detected in (True, True, False, False):
            phone.update(detected)

        self.assertFalse(phone.update(True))
        self.assertTrue(phone.update(True))

    def test_invalid_filter_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            DetectionPersistenceFilter(window_size=2, min_hits=3)


if __name__ == "__main__":
    unittest.main()
