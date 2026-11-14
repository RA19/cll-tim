# Implementation and reproducibility

[Overview](../README.md) · [Using CLL-TIM](USAGE.md) · [Evidence](EVIDENCE.md) · [Methods](METHODS.md)

## What this repository supplies

| Path | Purpose |
|---|---|
| `CLLTIM_MobileVer2.py` | Standalone multi-patient application; header dated 30 September 2025 |
| `CLLTIM_MobileVer1.py` | Historical application; its default single-patient input is not bundled |
| `CLLTIMData/` | 28 fitted learners, feature indexes/dictionaries, fitted scalers, imputation values and reporting assets |
| `PatientFiles/SynthPat_Multi.csv` | Nine synthetic patients |
| `PatientFileFormat.xlsx` | Input dictionary |
| `CLLTIMOutput/` | Generated reports; ignored by Git |
| `examples/verify_install.py` | Fresh-output synthetic run with strict reference comparison |
| `examples/clltim_showcase.py` | Learner-vote illustration, with no SHAP or missingness-performance claims |
| `docs/synthetic_patient_examples/` | Bundled reference reports, text and CSVs |
| `docs/papers/` | Original model paper, supplement and deployment paper |
| `docs/figures/` | Overview diagram, synthetic illustration and publication figures |
| `docs/talks/` | Historical Cambridge summer-school lecture |

The scoring applications and serialized model assets are preserved. The v2 header postdates the 2024 deployment paper; exact identity with the hospital's deployed release has not been established. The EHR integration described in that paper also required institution-specific mapping, automation and monitoring, which are not supplied by running the CSV application.

This is an inference/example repository, not a complete reproduction of training or the clinical studies. Real-patient study data are not included. Source/weight licensing has not been inferred from the papers' figure licenses.

## Scope cleanup

The project owner confirmed that the previously included `pipeline_scripts/` files came from COVIMUN work and that `Patient01.mp4` / `Patient02.mp4` were EEG material. Both unrelated groups have been removed from the current repository tree. They were not part of the CLL-TIM scoring path. Their removal does not rewrite Git history or alter originals held outside this repository.

## Feature explanations

The mobile application's score uses probability column 1. Its forest and linear SHAP branches select class index 0, while its XGBoost branch uses default margin-scale explanations. The application then sums these values into a common feature ranking. Class direction and output units therefore need independent validation before the signed aggregate can be interpreted as an explanation of the final probability.

A diagnostic on synthetic patient `SynthPat01`, using the pinned libraries, found:

| Example learner | Class-1 probability used for scoring | Selected explanation output |
|---|---|---|
| XGBoost BL0 | 0.589890 | Margin 0.363510; sigmoid gives 0.589890 |
| Random Forest BL1 | 0.620705 | Class-0 probability 0.379295 |
| Logistic Regression BL16 | 0.505580 | Class-0 probability 0.494420 |

These are implementation findings on the supplied mobile application. They do not establish that the original publications used the same explanation code. Matching the legacy reference reports verifies reproduction of that implementation; it does not independently validate the explanations.

The auxiliary showcase generator now illustrates learner outputs only. Its earlier SHAP and random-missingness demonstrations were removed because they did not provide a validated probability explanation or clinical robustness assessment. A future explanation correction should be versioned separately and checked against learner classes, output units and the ensemble aggregation.

## Synthetic verification

On 14 November 2026, a clean run on Windows with the existing pinned environment generated all 56 expected outputs. The nine text reports and two aggregate CSVs were byte-identical to the supplied references. The updated verifier was then exercised end to end: all nine patients completed and all 11 reference comparisons passed. Seven regression tests also passed, including missing references, changed contents, application failure and stale outputs. Neither the fitted models nor reference outputs were regenerated for this update.

The verifier now:

1. Checks that its input and comparison references exist and are nonempty.
2. Copies the runtime assets into a temporary working directory with empty outputs.
3. Runs the unmodified application and requires every expected file.
4. Compares text and semicolon-delimited CSV contents, failing on missing or differing references.
5. Copies verified outputs to `CLLTIMOutput/` after all comparisons pass.

CSV numeric comparisons use absolute tolerance `1e-6`; image/PDF checks require nonempty output files, without asserting binary identity. A failed check does not publish that run's outputs. Other pre-existing files in `CLLTIMOutput/` are left in place.

This demonstrates synthetic inference reproduction in the tested environment. A fresh Conda installation, other operating systems, retraining, real-patient validation and equivalence with a clinical deployment are outside that check.

To check the verification helper's failure handling without running the model:

```sh
python -m unittest discover -s tests -v
```

## Research context

The deployment paper reports that EHR predictions were labeled research-use only. See [evidence](EVIDENCE.md) for cohort-specific results and the distinction between prediction validation, trial selection and treatment benefit.
