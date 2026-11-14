# Using CLL-TIM

[Overview](../README.md) · [Evidence](EVIDENCE.md) · [Methods](METHODS.md) · [Implementation](IMPLEMENTATION.md)

## Install and run the synthetic example

You need Git and Conda. Run these commands from a terminal; Spyder is optional.

```sh
git clone https://github.com/RA19/cll-tim.git
cd cll-tim
conda env create -f environment-pipxgboost.yml
conda activate clltim_calc_env
python examples/verify_install.py
```

The alternative environment file installs XGBoost 0.82 through pip while retaining the legacy numeric-library pins. The original [environment.yml](../environment.yml) is retained for provenance. Use the pinned versions to load the serialized models; this is not a current-Python application.

The verifier runs the unmodified v2 application in a temporary directory with an empty output folder. It requires all expected outputs and matching reference results. Only after a successful comparison does it copy the outputs to `CLLTIMOutput/`. Existing reports cannot satisfy a missing output in that run.

The runtime was exercised on Windows using an existing environment with Python 3.7.9, NumPy 1.16.4, scikit-learn 0.21.2, XGBoost 0.82 and SHAP 0.28.5. This does not establish that a fresh Conda solve succeeds on every operating system.

To run without verification:

```sh
python CLLTIM_MobileVer2.py
```

Run from the repository root, where `CLLTIMData/`, `PatientFiles/` and `CLLTIMOutput/` are located. The default input is `PatientFiles/SynthPat_Multi.csv`.

## Expected outputs

The nine-patient example produces 56 files: six per patient and two aggregate CSVs.

| File | Contents |
|---|---|
| `<Patient>.pdf` / `.png` | Risk score, confidence category and contributing-factor display |
| `<Patient>HeatmapsTop.pdf` / `.png` | Legacy feature-attribution heatmaps |
| `<Patient>_Report.pdf` | Combined PDF report |
| `<Patient>_RiskFactors.txt` | Reported high- and low-risk factors |
| `SynthPat_Multi_Outputs_FeatureVectors.csv` | Calculated 228-feature vectors, separated by semicolons |
| `SynthPat_Multi_Outputs_RiskFactors.csv` | Scores, confidence categories and reported factors, separated by semicolons |

Verification compares the nine text files and both CSVs. CSV numeric cells use an absolute tolerance of `1e-6`; text fields must match after trimming surrounding whitespace. The text reports are compared after whitespace normalization. PDF and PNG files must be present and nonempty; their binary identity is not required.

## Patient-file format

Start with the [complete synthetic CSV](../PatientFiles/SynthPat_Multi.csv) and [PatientFileFormat.xlsx](../PatientFileFormat.xlsx). The workbook describes variable names and encodings; the mobile v2 application implements baseline, laboratory and infection-indicator inputs.

```csv
Patient,Diag_Date,Prediction_Point,Variable_Category,Variable_Name,Date_of_Test,Value_of_Test,Units_of_Test
TestPat01,2018-02-22,2018-05-10,lab,b2m,2018-04-19,5.7,mg_L
TestPat01,2018-02-22,2018-05-10,ii,infec,2017-06-18,0,na
TestPat01,2018-02-22,2018-05-10,baseline,ighv,na,unmu,na
```

These rows show the format; they are not a complete patient record.

| Column | Meaning |
|---|---|
| `Patient` | Patient identifier; also used in output filenames |
| `Diag_Date` | CLL diagnosis date in `YYYY-MM-DD` format |
| `Prediction_Point` | Date at which the prediction is made, in the same format |
| `Variable_Category` | `baseline`, `lab` or `ii` for this implementation |
| `Variable_Name` | Exact supported variable code from the workbook/example |
| `Date_of_Test` | Observation date, or the documented placeholder for baseline variables |
| `Value_of_Test` | Numeric value or supported categorical code |
| `Units_of_Test` | Unit code used by the example and dictionary |

Use consistent diagnosis and prediction dates within a patient's rows. Supply observations available by the prediction point. Convert measurements to the required units before creating the CSV; a unit label alone does not convert a value. Keep patient identifiers suitable for filenames.

Use the documented missing-value conventions. A missing observation is not automatically a measured zero. The mobile implementation does not read the original model's pathology and diagnosis-code histories; it fills the corresponding feature positions internally.

## Select a different input file

Place your CSV in `PatientFiles/`. Near the top of `CLLTIM_MobileVer2.py`, edit:

```python
PatientFileName = "YourPatients.csv"
```

If necessary, change `CLLTIM_PatientFolder` and `CLLTIM_OutputFolder` there too. The output directory must already exist. The application has no command-line input-file argument.

Keep `PPointInd="FromPatientFile"` to use the CSV's prediction point. The optional `Today` setting changes that date; it is not equivalent to the development study's prediction schedule. The application uses the finite date lookup in `CLLTIMData/Date2Int.csv`, so dates must be represented there.

Run the application directly after changing its configuration. The installation verifier is specifically for the unmodified default synthetic configuration.

## Reading a report

The score concerns the composite outcome of infection and/or CLL treatment within two years of the prediction point. In the original study, infection was represented by a blood-culture draw, independent of the culture result. Later prospective trial follow-up uses a different infection definition; see [evidence](EVIDENCE.md).

The mobile application rounds the combined score to two decimals. It labels a score above 0.5 high risk, and uses the following confidence categories:

| Rounded score | Report category |
|---|---|
| Below 0.28 | Low risk, high confidence |
| 0.28 through 0.50 | Low risk, low confidence |
| Above 0.50 and below 0.58 | High risk, low confidence |
| At least 0.58 | High risk, high confidence |

Confidence is derived from the mean score, not a separate calculation of vote variance. The score is not presented here as an independently established calibrated individual probability. The signed contributing factors have [documented implementation limitations](IMPLEMENTATION.md#feature-explanations).

## Troubleshooting

| Symptom | Check |
|---|---|
| Model loading or library errors | Activate the pinned environment and confirm the terminal/IDE uses its Python interpreter. |
| DLL-loading error on Windows | Use an activated Conda terminal so the environment's library directories are on its search path. |
| Crash in the old PyPDF2 warning formatter | The verifier sets `PYTHONWARNINGS=ignore`. For a direct run, use `$env:PYTHONWARNINGS='ignore'` in PowerShell, or `export PYTHONWARNINGS=ignore` in a POSIX shell. |
| Input or output path not found | Run from the repository root and check the configured folders and filename. |
| Date lookup/index error | Check date formatting and whether dates appear in `Date2Int.csv`. |
| Reference comparison fails | Inspect the named field and pinned versions. Do not refresh reference outputs merely to make a failed comparison pass. |
| v1 asks for `SynthPat01.csv` | v1 is historical and its default input is not bundled; use v2 for the documented example. |

The installed environment and matching synthetic outputs verify software reproduction. They do not repeat the real-patient validation studies.
