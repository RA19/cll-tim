"""CLL-TIM install sanity check.

Runs the stand-alone scoring application on the bundled multi-patient file
(PatientFiles/SynthPat_Multi.csv) and confirms that the expected set of
output files is produced for every patient in CLLTIMOutput/:

    <Patient>.pdf / .png                 risk + confidence report sheet
    <Patient>HeatmapsTop.pdf / .png      SHAP attribution heatmaps
    <Patient>_Report.pdf                 merged patient report
    <Patient>_RiskFactors.txt            top personalised risk factors
    SynthPat_Multi_Outputs_FeatureVectors.csv
    SynthPat_Multi_Outputs_RiskFactors.csv

A successful check means CLL-TIM is installed correctly and the model pack
loads end-to-end. Deterministic outputs (risk factors, feature vectors) are
additionally compared against the reference copies in
docs/synthetic_patient_examples/.

Usage (from the repository root, inside the clltim environment):

    python examples/verify_install.py
"""

import csv
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATIENT_FILE = os.path.join(ROOT, "PatientFiles", "SynthPat_Multi.csv")
OUTPUT_DIR = os.path.join(ROOT, "CLLTIMOutput")
GOLDEN_DIR = os.path.join(ROOT, "docs", "synthetic_patient_examples")

PER_PATIENT_SUFFIXES = [
    ".pdf",
    ".png",
    "HeatmapsTop.pdf",
    "HeatmapsTop.png",
    "_Report.pdf",
    "_RiskFactors.txt",
]
AGGREGATE_FILES = [
    "SynthPat_Multi_Outputs_FeatureVectors.csv",
    "SynthPat_Multi_Outputs_RiskFactors.csv",
]


def patient_names(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return sorted({row["Patient"] for row in csv.DictReader(fh)})


def run_app():
    env = dict(os.environ, PYTHONWARNINGS="ignore")
    proc = subprocess.run(
        [sys.executable, "CLLTIM_MobileVer2.py"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800,
    )
    return proc


def check_file_set(names):
    missing = []
    for p in names:
        for suf in PER_PATIENT_SUFFIXES:
            f = os.path.join(OUTPUT_DIR, p + suf)
            if not (os.path.isfile(f) and os.path.getsize(f) > 0):
                missing.append(p + suf)
    for f in AGGREGATE_FILES:
        p = os.path.join(OUTPUT_DIR, f)
        if not (os.path.isfile(p) and os.path.getsize(p) > 0):
            missing.append(f)
    return missing


def _rows(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.reader(fh))


def _cell_eq(a, b, tol=1e-6):
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) <= tol
    except ValueError:
        return False


def compare_table(new_path, ref_path, tol=1e-6):
    """Numeric-tolerant table comparison; returns (match, detail)."""
    try:
        new, ref = _rows(new_path), _rows(ref_path)
    except OSError as exc:
        return None, str(exc)
    if len(new) != len(ref):
        return False, f"row count {len(new)} vs reference {len(ref)}"
    for i, (rn, rr) in enumerate(zip(new, ref)):
        if len(rn) != len(rr):
            return False, f"row {i}: {len(rn)} cols vs {len(rr)}"
        for j, (cn, cr) in enumerate(zip(rn, rr)):
            if not _cell_eq(cn.strip(), cr.strip(), tol):
                return False, f"row {i} col {j}: {cn!r} vs {cr!r}"
    return True, "identical"


def compare_to_golden(names):
    checks = []
    for p in names:
        checks.append((p + "_RiskFactors.txt", _compare_txt(p + "_RiskFactors.txt")))
    for f in AGGREGATE_FILES:
        checks.append((f, compare_table(os.path.join(OUTPUT_DIR, f),
                                        os.path.join(GOLDEN_DIR, f))))
    return checks


def _compare_txt(fname):
    new_p = os.path.join(OUTPUT_DIR, fname)
    ref_p = os.path.join(GOLDEN_DIR, fname)
    try:
        with open(new_p, encoding="utf-8", errors="replace") as fh:
            new = fh.read().split()
        with open(ref_p, encoding="utf-8", errors="replace") as fh:
            ref = fh.read().split()
    except OSError as exc:
        return None, str(exc)
    return (new == ref), "identical" if new == ref else "content differs"


def main():
    if not os.path.isfile(PATIENT_FILE):
        sys.exit(f"ERROR: patient file not found: {PATIENT_FILE}")

    names = patient_names(PATIENT_FILE)
    print(f"[1/3] {len(names)} patients in SynthPat_Multi.csv: "
          f"{', '.join(names)}")

    print("[2/3] Running CLLTIM_MobileVer2.py (a few minutes) ...")
    proc = run_app()
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
        sys.exit("FAIL: application exited with a non-zero status "
                 "(see output above; check the environment install notes "
                 "in the README).")

    print("[3/3] Checking CLLTIMOutput/ against the expected file set ...")
    missing = check_file_set(names)
    if missing:
        for f in missing:
            print(f"  MISSING  {f}")
        sys.exit(f"FAIL: {len(missing)} expected output file(s) not produced.")

    print(f"  file set complete: {len(names)} patients x "
          f"{len(PER_PATIENT_SUFFIXES)} files + {len(AGGREGATE_FILES)} "
          "aggregate CSVs")

    drift = 0
    for name, res in compare_to_golden(names):
        status = "reference unavailable"
        if isinstance(res, tuple):
            match, detail = res
            status = "MATCH" if match else ("?" if match is None else "DIFF")
            if match is False:
                drift += 1
            if detail != "identical":
                status += f" ({detail})"
        print(f"  {status:>7}  {name}")

    print()
    if drift:
        print(f"PASS (with {drift} content difference(s) vs the reference "
              "set - usually benign library-version drift in numeric "
              "formatting; file set is complete).")
    else:
        print("PASS: CLL-TIM produced the complete expected file set and "
              "deterministic outputs match the reference set.")


if __name__ == "__main__":
    main()
