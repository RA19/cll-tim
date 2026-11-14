"""Extend CLL-TIM's date lookup while preserving its existing byte prefix.

The existing table defines F1 as YYYY-MM-DD and F3 as DD-MM-YY. Infer and
validate F2's increasing or capped sequence from the complete input table.
Only new daily rows are generated; existing date codes are never rewritten.
"""

import argparse
import csv
from datetime import date, timedelta
import io
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "CLLTIMData" / "Date2Int.csv"


def extend_lookup(path, through):
    path = Path(path)
    original = path.read_bytes()
    encoding = "utf-8-sig" if original.startswith(b"\xef\xbb\xbf") else "utf-8"
    text = original.decode(encoding)
    newline = "\r\n" if "\r\n" in text else "\n"
    if newline == "\r\n" and "\n" in text.replace("\r\n", ""):
        raise ValueError("Mixed line endings in the existing lookup")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != ["F1", "F2", "F3"]:
        raise ValueError("Expected the existing F1,F2,F3 CSV schema")
    rows = list(reader)
    if len(rows) < 2:
        raise ValueError("At least two existing rows are needed to validate the pattern")

    first = date.fromisoformat(rows[0]["F1"])
    first_code = int(rows[0]["F2"])
    codes = [int(row["F2"]) for row in rows]
    differences = [right - left for left, right in zip(codes, codes[1:])]
    capped = 0 in differences
    cap = max(codes) if capped else None
    for index, row in enumerate(rows):
        day = first + timedelta(days=index)
        expected_code = min(first_code + index, cap) if capped else first_code + index
        if row != {"F1": day.isoformat(), "F2": str(expected_code),
                   "F3": day.strftime("%d-%m-%y")}:
            raise ValueError("Existing row %d does not follow the date/code pattern" % (index + 2))

    previous_end = first + timedelta(days=len(rows) - 1)
    result = {"previous_end": previous_end.isoformat(), "through": previous_end.isoformat(),
              "added_rows": 0, "f2_cap": cap}
    if through <= previous_end:
        return result

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator=newline)
    day = previous_end + timedelta(days=1)
    while day <= through:
        index = (day - first).days
        code = min(first_code + index, cap) if capped else first_code + index
        writer.writerow((day.isoformat(), str(code), day.strftime("%d-%m-%y")))
        day += timedelta(days=1)
    boundary = b"" if original.endswith(newline.encode("ascii")) else newline.encode("ascii")
    appended = output.getvalue().encode("utf-8")
    path.write_bytes(original + boundary + appended)
    result.update(through=through.isoformat(), added_rows=(through - previous_end).days)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--through", type=date.fromisoformat, default=date(2040, 12, 31),
                        help="Last supported date, inclusive (default: 2040-12-31)")
    args = parser.parse_args()
    result = extend_lookup(args.path, args.through)
    print("Added {added_rows} daily rows: {previous_end} -> {through}; F2 cap: {f2_cap}".format(**result))


if __name__ == "__main__":
    main()
