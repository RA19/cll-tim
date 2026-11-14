"""Date-lookup compatibility: unchanged prefix, leap days and capped codes."""

import csv
from datetime import date, timedelta
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "extend_date_lookup", Path(__file__).resolve().parents[1] / "scripts/extend_date_lookup.py",
)
lookup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lookup)


class DateLookupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="clltim-date-test-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "Date2Int.csv"

    def make_table(self, start, count, cap=None, newline="\r\n"):
        text = io.StringIO(newline="")
        writer = csv.writer(text, lineterminator=newline)
        writer.writerow(("F1", "F2", "F3"))
        for i in range(count):
            day = start + timedelta(days=i)
            writer.writerow((day.isoformat(), min(i + 1, cap) if cap else i + 1,
                             day.strftime("%d-%m-%y")))
        self.path.write_bytes(text.getvalue().encode("utf-8"))

    def test_capped_code_and_original_bytes_are_preserved(self):
        self.make_table(date(2039, 12, 27), 5, cap=3)
        original = self.path.read_bytes()
        result = lookup.extend_lookup(self.path, date(2040, 12, 31))
        updated = self.path.read_bytes()
        self.assertTrue(updated.startswith(original))
        self.assertEqual(result["added_rows"], 366)
        self.assertEqual(updated.count(b"\r\n"), updated.count(b"\n"))
        rows = list(csv.DictReader(updated.decode().splitlines()))
        self.assertEqual(rows[-1], {"F1": "2040-12-31", "F2": "3", "F3": "31-12-40"})
        self.assertEqual(sum(row["F1"] == "2040-02-29" for row in rows), 1)

    def test_second_run_does_not_change_the_file(self):
        self.make_table(date(2040, 1, 1), 5, cap=3)
        lookup.extend_lookup(self.path, date(2040, 12, 31))
        original = self.path.read_bytes()
        result = lookup.extend_lookup(self.path, date(2040, 12, 31))
        self.assertEqual(result["added_rows"], 0)
        self.assertEqual(self.path.read_bytes(), original)

    def test_uncapped_input_continues_its_sequence_and_lf_format(self):
        self.make_table(date(2032, 2, 27), 2, newline="\n")
        lookup.extend_lookup(self.path, date(2032, 3, 1))
        self.assertNotIn(b"\r", self.path.read_bytes())
        rows = list(csv.DictReader(self.path.read_text().splitlines()))
        self.assertEqual(rows[-2], {"F1": "2032-02-29", "F2": "3", "F3": "29-02-32"})
        self.assertEqual(rows[-1]["F2"], "4")

    def test_invalid_existing_pattern_is_rejected_without_writing(self):
        self.make_table(date(2040, 1, 1), 5, cap=3)
        original = self.path.read_bytes().replace(b"2040-01-03,3", b"2040-01-03,2")
        self.path.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "does not follow"):
            lookup.extend_lookup(self.path, date(2040, 12, 31))
        self.assertEqual(self.path.read_bytes(), original)

    def test_gap_in_existing_dates_is_rejected(self):
        self.make_table(date(2040, 1, 1), 5)
        self.path.write_bytes(self.path.read_bytes().replace(b"2040-01-03", b"2040-01-04"))
        with self.assertRaisesRegex(ValueError, "does not follow"):
            lookup.extend_lookup(self.path, date(2040, 12, 31))


if __name__ == "__main__":
    unittest.main()
