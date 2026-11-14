# Using CLL-TIM

[Overview](../README.md) · [Validation](EVIDENCE.md) · [Methods](METHODS.md) · [Technical reference](IMPLEMENTATION.md)

CLL-TIM predicts the two-year risk of infection and/or CLL treatment and provides predictive confidence and personalized risk factors for each patient. Begin with the [complete synthetic patient report](synthetic_patient_examples/SynthPat01_Report.pdf), then run the bundled examples or prepare your own patient CSV.

## Install CLL-TIM

From a terminal with Git and Conda available:

```sh
git clone https://github.com/RA19/cll-tim.git
cd cll-tim
conda env create -f environment-pipxgboost.yml
conda activate clltim_calc_env
```

The environment specifies Python 3.7.9 and the library versions used to load the trained models. XGBoost 0.82 is installed through pip. The original [environment.yml](../environment.yml) is also available. If you use an IDE, select the Python interpreter inside `clltim_calc_env`.

## Run CLL-TIM

```sh
python CLLTIM_MobileVer2.py
```

Run from the repository root. By default, CLL-TIM reads `PatientFiles/SynthPat_Multi.csv`, containing nine synthetic patients, and writes reports to `CLLTIMOutput/`.

To run the examples and compare their results with the bundled references:

```sh
python examples/verify_install.py
```

The verifier uses fresh outputs and copies reports to `CLLTIMOutput/` after the comparisons pass. Use the default synthetic input configuration for this installation check.

## Patient-file format

Use the [complete example CSV](../PatientFiles/SynthPat_Multi.csv) and [PatientFileFormat.xlsx](../PatientFileFormat.xlsx) as templates. Each row represents an observation. Repeat the patient identifier across that patient's rows; a single CSV may contain several patients.

```csv
Patient,Diag_Date,Prediction_Point,Variable_Category,Variable_Name,Date_of_Test,Value_of_Test,Units_of_Test
TestPat01,2018-02-22,2018-05-10,lab,b2m,2018-04-19,5.7,mg_L
TestPat01,2018-02-22,2018-05-10,ii,infec,2017-06-18,0,na
TestPat01,2018-02-22,2018-05-10,baseline,ighv,na,unmu,na
```

These rows illustrate the format. The complete example contains additional clinical markers, laboratory results and infection history.

| Column | Required information |
|---|---|
| `Patient` | Patient identifier, also used to name the reports |
| `Diag_Date` | CLL diagnosis date, `YYYY-MM-DD` |
| `Prediction_Point` | Date from which the two-year outcome window begins, `YYYY-MM-DD` |
| `Variable_Category` | `baseline` for CLL markers, `lab` for laboratory results, or `ii` for infection indicators |
| `Variable_Name` | Exact variable code from the dictionary, such as `b2m`, `ighv` or `infec` |
| `Date_of_Test` | Observation date, or the documented placeholder for baseline variables |
| `Value_of_Test` | Measurement or supported categorical value |
| `Units_of_Test` | Unit code specified in the dictionary |

Keep diagnosis and prediction dates consistent within a patient's rows. Include observations available by the prediction point, and convert measurements to the required units before creating the CSV. A unit label alone does not convert the value. Use the documented missing-value conventions; retain measured zeros where appropriate.

The scoring script processes baseline CLL markers, laboratory results and infection indicators. Pathology and diagnosis-code histories are not read by this script; their feature positions are filled internally. Date coverage extends from **1 January 1980 through 31 December 2040**.

## Select your patient file

Place your CSV in `PatientFiles/`. Near the top of `CLLTIM_MobileVer2.py`, set:

```python
PatientFileName = "YourPatients.csv"
```

If required, change `CLLTIM_PatientFolder` and `CLLTIM_OutputFolder` there too. Ensure that the output directory exists. Then run:

```sh
python CLLTIM_MobileVer2.py
```

Keep `PPointInd="FromPatientFile"` to use each patient's prediction point. The optional `Today` setting uses the current date instead. Our development prediction point was three months after diagnosis; changing this timing changes the clinical context of prediction.

## Interpreting the patient report

### Two-year risk of infection and/or CLL treatment

CLL-TIM estimates risk of the composite endpoint over the two years following the prediction point. The report classifies the patient as high or low risk. In our original cohorts, infection was represented by a blood-culture draw, irrespective of the result. PreVent-ACaLL follow-up uses CTCAE grade ≥3 infection. [Clinical endpoints and validation results](EVIDENCE.md).

### Predictive confidence

Predictive confidence indicates how certain CLL-TIM is about the individual prediction. We derived high-confidence thresholds to identify patients with more reliable high- and low-risk predictions. CLL-TIM still provides a prediction for low-confidence patients, with greater uncertainty.

The scoring script rounds the ensemble's mean prediction to two decimals and assigns:

| Rounded prediction | Risk and predictive confidence |
|---|---|
| Below 0.28 | Low risk, high confidence |
| 0.28 through 0.50 | Low risk, low confidence |
| Above 0.50 and below 0.58 | High risk, low confidence |
| At least 0.58 | High risk, high confidence |

The underlying calculation and validation of predictive confidence are described in the [methods](METHODS.md#predictive-confidence).

### Personalized risk factors

The report lists the five factors driving the individual prediction towards higher risk and the five driving it towards lower risk. These help relate the prediction to the patient's clinical and laboratory history. Patients with similar risk predictions can have different personalized risk factors.

We use SHAP for these explanations. See the [technical reference](IMPLEMENTATION.md#personalized-risk-factors) for the class and output-scale conventions in the scoring script.

## Report files

CLL-TIM writes six files per patient, plus two aggregate CSVs for the input file. The nine-patient example therefore produces 56 files.

| File | Contents |
|---|---|
| `<Patient>.pdf` / `.png` | Two-year infection/treatment risk, predictive confidence and personalized risk factors |
| `<Patient>HeatmapsTop.pdf` / `.png` | Personalized risk-factor heatmaps |
| `<Patient>_Report.pdf` | Combined patient report |
| `<Patient>_RiskFactors.txt` | Personalized high- and low-risk factors in text format |
| `<InputName>_Outputs_FeatureVectors.csv` | Calculated 228-feature vectors |
| `<InputName>_Outputs_RiskFactors.csv` | Risk predictions, predictive confidence and personalized risk factors |

The aggregate CSVs use semicolons as separators.

## Troubleshooting

| Symptom | Action |
|---|---|
| Model-loading or library error | Activate `clltim_calc_env` and use the pinned library versions. |
| DLL-loading error on Windows | Run from an activated Conda terminal so its library directories are available. |
| Error in the old PyPDF2 warning formatter | Set `$env:PYTHONWARNINGS='ignore'` in PowerShell, or `export PYTHONWARNINGS=ignore` in a POSIX shell. The verifier sets this automatically. |
| Input/output path not found | Run from the repository root and check the configured filenames and folders. |
| Date lookup error | Use `YYYY-MM-DD` and dates within the supported range. See [date lookup](IMPLEMENTATION.md#date-lookup) for the extension script. |
| Reference comparison fails | Inspect the named field and confirm the pinned environment and default synthetic input configuration. |
