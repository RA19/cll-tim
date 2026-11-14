# Technical reference

[Overview](../README.md) · [Using CLL-TIM](USAGE.md) · [Validation](EVIDENCE.md) · [Methods](METHODS.md)

## Repository contents

| Path | Purpose |
|---|---|
| `CLLTIM_MobileVer2.py` | CLL-TIM scoring script for one or more patients |
| `CLLTIMData/` | Trained learners, feature definitions, fitted transformations and date lookup |
| `PatientFiles/SynthPat_Multi.csv` | Nine synthetic patient records |
| `PatientFileFormat.xlsx` | Input variable definitions and encodings |
| `CLLTIMOutput/` | Generated patient reports and CSV results |
| `examples/verify_install.py` | Run the synthetic patients and compare results with the reference outputs |
| `examples/clltim_showcase.py` | Generate the synthetic learner-prediction illustration |
| `scripts/extend_date_lookup.py` | Extend the date lookup using its existing formats and numeric-code sequence |
| `docs/synthetic_patient_examples/` | Synthetic patient reports and reference results |
| `docs/papers/` | Our model and deployment publications, including the 2020 supplement |
| `docs/figures/` | CLL-TIM overview, publication figures and synthetic illustration |
| `docs/talks/` | Cambridge summer-school lecture |

The repository supports inference with the trained CLL-TIM ensemble. Clinical study data and a complete executable training workflow are not included. Our [deployment methods](EVIDENCE.md#ehr-deployment-and-validation--2024) describe the additional data harmonization, EHR integration and monitoring used in clinical research.

## Model assets and runtime

`CLLTIMData/CLL-TIM_BLN_*.sav` contains the 28 fitted learners. Feature-index arrays specify the inputs to each learner; the pickle files hold fitted imputers, scalers and SHAP background summaries. Use the versions in [environment-pipxgboost.yml](../environment-pipxgboost.yml) to load these serialized objects.

CLL-TIM constructs 228 features from the patient CSV. Non-XGBoost learners use the stored imputation values and fitted scalers. XGBoost handles missing values internally. Learner probabilities are averaged and rounded to two decimals for reporting. [Model methods](METHODS.md).

## Date lookup

`CLLTIMData/Date2Int.csv` covers **1 January 1980 through 31 December 2040**, inclusive. Its columns are:

| Column | Format |
|---|---|
| `F1` | Calendar date, `YYYY-MM-DD` |
| `F2` | Existing numeric-code sequence: increasing initially, then capped at its historical maximum |
| `F3` | Calendar date, `DD-MM-YY` |

CLL-TIM uses the row position of a matching `F1` date for date calculations. Existing rows must retain their order and values. To extend coverage, run:

```sh
python scripts/extend_date_lookup.py --through 2040-12-31
```

The script validates all existing rows, infers the numeric-code pattern, and appends consecutive dates with the same date formats and line endings. Running it again for an already covered date leaves the file unchanged.

## Personalized risk factors

We use SHAP to derive personalized risk factors, as described in our [model methods](METHODS.md#personalized-risk-factors). The scoring script selects class-0 SHAP values for forest and linear learners and margin-scale SHAP values for XGBoost, while risk predictions use class-1 probabilities. These different output conventions affect the direction and scale of aggregated signed contributions. The aggregate should therefore not be interpreted as an additive decomposition of the final probability. The risk-prediction and SHAP calculations follow separate code paths.

## Installation verification

```sh
python examples/verify_install.py
```

The verifier runs CLL-TIM on the bundled synthetic patients in a temporary directory with empty outputs. It requires all expected files and matching reference text/CSV results before copying reports to `CLLTIMOutput/`. Missing references or content differences fail the check.

CSV numeric comparisons use absolute tolerance `1e-6`. PDF and PNG files must be nonempty. Reference comparison checks reproducibility of the scoring script; clinical performance is assessed in the [validation cohorts](EVIDENCE.md).

Developer tests for date-lookup compatibility and verification behavior:

```sh
python -m unittest discover -s tests -v
```
