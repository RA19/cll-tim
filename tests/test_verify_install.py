"""Regression checks for false-success cases in the synthetic verifier."""

import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "verify_install", Path(__file__).resolve().parents[1] / "examples/verify_install.py",
)
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="clltim-verifier-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "CLLTIMOutput"
        self.golden = self.root / "reference"
        self.patient_file = self.root / "PatientFiles/SynthPat_Multi.csv"
        for folder in (self.output, self.golden, self.patient_file.parent, self.root / "CLLTIMData"):
            folder.mkdir()
        self.patient_file.write_text("Patient\nExample01\n", encoding="utf-8")
        (self.root / "CLLTIM_MobileVer2.py").write_text("# stub\n", encoding="utf-8")
        for name in verify.reference_files(["Example01"]):
            (self.golden / name).write_text(
                "Feature;Example01\nAge;0.67\n" if name.endswith(".csv") else "reference factors\n",
                encoding="utf-8",
            )
        patcher = mock.patch.multiple(verify, ROOT=str(self.root), OUTPUT_DIR=str(self.output),
                                     GOLDEN_DIR=str(self.golden), PATIENT_FILE=str(self.patient_file))
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_main(self, run_app):
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(verify, "run_app", side_effect=run_app):
            return verify.main()

    def generate(self, workspace):
        out = Path(workspace) / "CLLTIMOutput"
        for name in verify.expected_files(["Example01"]):
            reference = self.golden / name
            (out / name).write_bytes(reference.read_bytes() if reference.exists() else b"fresh output")
        return subprocess.CompletedProcess([], 0, "", "")

    def test_stale_outputs_cannot_satisfy_a_fresh_run(self):
        for name in verify.expected_files(["Example01"]):
            (self.output / name).write_text("stale", encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "expected outputs not produced"):
            self.run_main(lambda _: subprocess.CompletedProcess([], 0, "", ""))
        self.assertEqual((self.output / "Example01.pdf").read_text(), "stale")

    def test_missing_reference_fails_before_execution(self):
        (self.golden / "Example01_RiskFactors.txt").unlink()
        with mock.patch.object(verify, "run_app") as run:
            with self.assertRaisesRegex(SystemExit, "missing or empty reference"):
                verify.main()
            run.assert_not_called()

    def test_changed_reference_content_fails_and_does_not_copy_outputs(self):
        def changed(workspace):
            result = self.generate(workspace)
            (Path(workspace) / "CLLTIMOutput/Example01_RiskFactors.txt").write_text("different")
            return result
        with self.assertRaisesRegex(SystemExit, "reference comparison.*differ"):
            self.run_main(changed)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_matching_fresh_run_copies_verified_files(self):
        (self.output / "unrelated-report.txt").write_text("keep")
        self.run_main(self.generate)
        for name in verify.expected_files(["Example01"]):
            self.assertTrue((self.output / name).is_file())
        self.assertEqual((self.output / "unrelated-report.txt").read_text(), "keep")
        self.assertFalse(list(self.output.glob(".verify-*")))

    def test_semicolon_csv_numeric_tolerance_is_applied_per_cell(self):
        actual, reference = self.root / "actual.csv", self.root / "expected.csv"
        reference.write_text("patient;score\nA;0.670000\n")
        actual.write_text("patient;score\nA;0.6700005\n")
        self.assertTrue(verify.compare_table(str(actual), str(reference))[0])
        actual.write_text("patient;score\nA;0.670005\n")
        self.assertFalse(verify.compare_table(str(actual), str(reference))[0])

    def test_csv_category_changes_are_not_numeric_matches(self):
        self.assertFalse(verify._cell_eq("High", "Low"))
        self.assertFalse(verify._cell_eq("", "0"))

    def test_application_failure_does_not_copy_outputs(self):
        with self.assertRaisesRegex(SystemExit, "non-zero status"):
            self.run_main(lambda _: subprocess.CompletedProcess([], 1, "", "failure"))
        self.assertEqual(list(self.output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
