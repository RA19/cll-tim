"""Verify a fresh synthetic run, then copy verified reports to CLLTIMOutput/.

Runs CLL-TIM in an isolated working directory. Missing
outputs/references and differences in reference text/CSV contents fail the check.
This checks reproduction of CLL-TIM results, not clinical or SHAP correctness.
Use the pinned runtime: python examples/verify_install.py
"""

import csv
import math
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATIENT_FILE = os.path.join(ROOT, "PatientFiles", "SynthPat_Multi.csv")
OUTPUT_DIR = os.path.join(ROOT, "CLLTIMOutput")
GOLDEN_DIR = os.path.join(ROOT, "docs", "synthetic_patient_examples")

PER_PATIENT_SUFFIXES = (
    ".pdf", ".png", "HeatmapsTop.pdf", "HeatmapsTop.png",
    "_Report.pdf", "_RiskFactors.txt",
)
AGGREGATE_FILES = (
    "SynthPat_Multi_Outputs_FeatureVectors.csv",
    "SynthPat_Multi_Outputs_RiskFactors.csv",
)


def patient_names(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return sorted({row["Patient"] for row in csv.DictReader(handle)})


def expected_files(names):
    return [p + suffix for p in names for suffix in PER_PATIENT_SUFFIXES] + list(AGGREGATE_FILES)


def reference_files(names):
    return [p + "_RiskFactors.txt" for p in names] + list(AGGREGATE_FILES)


def missing_files(directory, files):
    return [name for name in files if not (
        os.path.isfile(os.path.join(directory, name))
        and os.path.getsize(os.path.join(directory, name)) > 0
    )]


def run_clltim(workspace):
    env = dict(os.environ, PYTHONWARNINGS="ignore", MPLBACKEND="Agg")
    return subprocess.run(
        [sys.executable, "CLLTIM_MobileVer2.py"],
        cwd=workspace, env=env, capture_output=True, text=True, timeout=1800,
    )


def _rows(path):
    # Both CLL-TIM aggregate CSVs are semicolon-delimited.
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.reader(handle, delimiter=";"))


def _cell_eq(a, b, tol=1e-6):
    a, b = a.strip(), b.strip()
    if a == b:
        return True
    try:
        left, right = float(a), float(b)
    except ValueError:
        return False
    return (math.isnan(left) and math.isnan(right)) or math.isclose(
        left, right, rel_tol=0, abs_tol=tol,
    )


def compare_table(new_path, ref_path, tol=1e-6):
    try:
        new, ref = _rows(new_path), _rows(ref_path)
    except OSError as exc:
        return False, str(exc)
    if len(new) != len(ref):
        return False, "row count %d vs reference %d" % (len(new), len(ref))
    for i, (rn, rr) in enumerate(zip(new, ref)):
        if len(rn) != len(rr):
            return False, "row %d: %d columns vs reference %d" % (i, len(rn), len(rr))
        for j, (cn, cr) in enumerate(zip(rn, rr)):
            if not _cell_eq(cn, cr, tol):
                return False, "row %d column %d: %r vs %r" % (i, j, cn, cr)
    return True, "match"


def compare_to_golden(names, output_dir):
    checks = []
    for name in reference_files(names):
        new_path = os.path.join(output_dir, name)
        ref_path = os.path.join(GOLDEN_DIR, name)
        if name.endswith(".csv"):
            result = compare_table(new_path, ref_path)
        else:
            try:
                with open(new_path, encoding="utf-8") as handle:
                    new = handle.read().split()
                with open(ref_path, encoding="utf-8") as handle:
                    ref = handle.read().split()
                result = (new == ref, "match" if new == ref else "content differs")
            except (OSError, UnicodeError) as exc:
                result = (False, str(exc))
        checks.append((name, result))
    return checks


def main():
    if not os.path.isfile(PATIENT_FILE):
        sys.exit("FAIL: patient file not found: " + PATIENT_FILE)
    names = patient_names(PATIENT_FILE)
    if not names:
        sys.exit("FAIL: no synthetic patients found.")
    unavailable = missing_files(GOLDEN_DIR, reference_files(names))
    if unavailable:
        sys.exit("FAIL: missing or empty reference files: " + ", ".join(unavailable))

    print("[1/3] %d patients in SynthPat_Multi.csv" % len(names), flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_root = os.path.realpath(OUTPUT_DIR)
    with tempfile.TemporaryDirectory(prefix=".verify-", dir=output_root) as workspace:
        # Cleanup is confined to a newly created directory inside CLLTIMOutput.
        if os.path.commonpath([output_root, os.path.realpath(workspace)]) != output_root:
            sys.exit("FAIL: verification directory is outside the output folder.")
        shutil.copy2(os.path.join(ROOT, "CLLTIM_MobileVer2.py"), workspace)
        shutil.copytree(os.path.join(ROOT, "CLLTIMData"), os.path.join(workspace, "CLLTIMData"))
        os.mkdir(os.path.join(workspace, "PatientFiles"))
        shutil.copy2(PATIENT_FILE, os.path.join(workspace, "PatientFiles", "SynthPat_Multi.csv"))
        fresh_output = os.path.join(workspace, "CLLTIMOutput")
        os.mkdir(fresh_output)

        print("[2/3] Running CLL-TIM with empty outputs (a few minutes) ...", flush=True)
        try:
            proc = run_clltim(workspace)
        except subprocess.TimeoutExpired:
            sys.exit("FAIL: CLL-TIM exceeded the 30-minute verification timeout.")
        if proc.returncode != 0:
            print(proc.stdout[-4000:])
            print(proc.stderr[-4000:])
            sys.exit("FAIL: CLL-TIM exited with non-zero status; see docs/USAGE.md.")

        print("[3/3] Checking fresh outputs and reference contents ...", flush=True)
        missing = missing_files(fresh_output, expected_files(names))
        if missing:
            sys.exit("FAIL: expected outputs not produced: " + ", ".join(missing))
        failed = []
        for name, (match, detail) in compare_to_golden(names, fresh_output):
            print("  %s  %s%s" % (
                "MATCH" if match else "FAIL", name,
                "" if match else " (" + detail + ")",
            ))
            if not match:
                failed.append(name)
        if failed:
            sys.exit("FAIL: %d reference comparison(s) differ; no outputs copied." % len(failed))

        for name in expected_files(names):
            shutil.copy2(os.path.join(fresh_output, name), os.path.join(output_root, name))

    print("PASS: %d fresh outputs; %d reference comparisons match. Reports: %s" % (
        len(expected_files(names)), len(reference_files(names)), OUTPUT_DIR,
    ))


if __name__ == "__main__":
    main()
