import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "01_prepare_nthu_manifest.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_nthu_manifest", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ParseNthuPathTests(unittest.TestCase):
    def test_parses_standard_awake_filename(self):
        root = Path("dataset")
        path = root / "frames" / "005_glasses_yawning_162_notdrowsy.jpg"

        record, reason = MODULE.parse_nthu_path(path, root)

        self.assertIsNone(reason)
        self.assertEqual(record.subject_id, "005")
        self.assertEqual(record.frame_index, 162)
        self.assertEqual(record.scenario, "glasses_yawning")
        self.assertEqual(record.label, "awake")
        self.assertEqual(record.label_id, 0)
        self.assertEqual(record.video_id, "005_glasses_yawning_notdrowsy")

    def test_parses_standard_drowsy_filename(self):
        root = Path("dataset")
        path = root / "001_noglasses_sleepyCombination_42_drowsy.jpg"

        record, reason = MODULE.parse_nthu_path(path, root)

        self.assertIsNone(reason)
        self.assertEqual(record.label, "drowsy")
        self.assertEqual(record.label_id, 1)
        self.assertEqual(record.frame_index, 42)

    def test_rejects_unknown_subject(self):
        root = Path("dataset")
        path = root / "driverA_glasses_yawning_10_drowsy.jpg"

        record, reason = MODULE.parse_nthu_path(path, root)

        self.assertIsNone(record)
        self.assertEqual(reason, "subject_id_is_not_three_digits")


class SubjectSplitTests(unittest.TestCase):
    def test_four_subjects_become_two_one_one(self):
        assignment = MODULE.assign_subject_splits(
            ["001", "002", "003", "004"], seed=42
        )
        counts = {
            split: list(assignment.values()).count(split)
            for split in ("train", "val", "test")
        }
        self.assertEqual(counts, {"train": 2, "val": 1, "test": 1})

    def test_split_is_deterministic(self):
        subjects = ["001", "002", "003", "004", "005"]
        first = MODULE.assign_subject_splits(subjects, seed=7)
        second = MODULE.assign_subject_splits(reversed(subjects), seed=7)
        self.assertEqual(first, second)

    def test_requires_three_subjects(self):
        with self.assertRaises(ValueError):
            MODULE.assign_subject_splits(["001", "002"])


class EndToEndManifestTests(unittest.TestCase):
    def test_scans_splits_audits_and_saves_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "nthu"
            output = Path(temp_dir) / "outputs"
            root.mkdir()
            for subject in ("001", "002", "003", "004"):
                for frame_index, label in ((1, "notdrowsy"), (2, "drowsy")):
                    filename = (
                        f"{subject}_glasses_yawning_{frame_index}_{label}.jpg"
                    )
                    (root / filename).touch()

            records, skipped = MODULE.scan_dataset(root)
            assignment = MODULE.assign_subject_splits(
                (record.subject_id for record in records), seed=42
            )
            split_records = MODULE.apply_splits(records, assignment)
            summary = MODULE.audit_manifest(split_records)
            MODULE.save_outputs(
                output,
                root,
                split_records,
                skipped,
                assignment,
                summary,
                seed=42,
            )

            self.assertEqual(len(records), 8)
            self.assertEqual(skipped, [])
            self.assertEqual(summary["status"], "PASS - subject sets are disjoint")
            self.assertTrue((output / "nthu_manifest.csv").is_file())
            self.assertTrue((output / "subject_split.json").is_file())
            saved_split = json.loads(
                (output / "subject_split.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_split["assignment"], assignment)


if __name__ == "__main__":
    unittest.main()
