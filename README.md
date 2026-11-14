# CLL-TIM

### Predicting infection and treatment risk in newly diagnosed chronic lymphocytic leukaemia

We developed the **CLL Treatment–Infection Model (CLL-TIM)** to identify newly diagnosed patients with chronic lymphocytic leukaemia (CLL) at high risk of **infection and/or initiation of CLL treatment within two years**. CLL-TIM addresses an unmet clinical need during watchful waiting: identifying patients at increased risk of infection or early treatment, including those who do not yet meet iwCLL treatment criteria.

CLL-TIM integrates baseline prognostic markers with longitudinal laboratory results and infection histories extending up to seven years. An ensemble of 28 machine-learning algorithms combines these data to provide three patient-specific outputs: **predicted risk of infection and/or CLL treatment, predictive confidence, and personalized risk factors** explaining the patient's prediction.

We developed and internally validated CLL-TIM in Denmark, externally validated it in the German CLL7 cohort, and prospectively validated it through **PreVent-ACaLL across Denmark, Sweden and the Netherlands**. We also integrated CLL-TIM into an Epic-based Danish electronic health record (EHR). Independent investigators subsequently reported validation in a US cohort. The [validation guide](docs/EVIDENCE.md) describes the cohorts, clinical endpoints and results.

[Clinical purpose](#why-we-developed-cll-tim) · [Validation and use](#where-we-validated-and-used-cll-tim) · [Run CLL-TIM](#run-cll-tim) · [Patient data](docs/USAGE.md#patient-file-format) · [Methods](docs/METHODS.md)

![CLL-TIM combines clinical markers, laboratory results and infection histories to predict the two-year risk of infection and/or CLL treatment, with predictive confidence and personalized risk factors.](docs/figures/overview.svg)

## Why we developed CLL-TIM

Patients with CLL can experience serious infections before they require treatment. Identifying these patients at diagnosis is important for studying strategies to prevent infection and improve outcomes. We therefore modeled infection and CLL treatment together, capturing both the infectious burden associated with CLL-related immune dysfunction and the risk of subsequent treatment.

We designed CLL-TIM to work with the heterogeneous, incomplete longitudinal records encountered in clinical practice. Alongside each prediction, predictive confidence and personalized risk factors help physicians assess its reliability and understand which aspects of the patient's history drive the predicted risk.

## What CLL-TIM provides for each patient

| Output | Clinical interpretation |
|---|---|
| **Two-year risk of infection and/or CLL treatment** | The predicted risk of the composite outcome within two years of the prediction point, with classification as high or low risk. |
| **Predictive confidence** | How certain CLL-TIM is about that patient's prediction. High-confidence predictions identify patients for whom CLL-TIM has greater certainty; low-confidence predictions indicate greater uncertainty. |
| **Personalized risk factors** | The factors driving the individual prediction towards higher or lower risk, derived using SHAP. These can differ between patients with similar predicted risk. |

In our original development and validation, we used **a blood-culture draw as a proxy for infection**, irrespective of the culture result. The development prediction point was three months after CLL diagnosis, and outcomes were assessed over the following two years. The prospective PreVent-ACaLL analysis uses CTCAE grade ≥3 infection. See [endpoint definitions](docs/EVIDENCE.md#clinical-endpoint-and-prediction-point) and [how to interpret a report](docs/USAGE.md#interpreting-the-patient-report).

## Where we validated and used CLL-TIM

| Study | Clinical setting | Main findings and use |
|---|---|---|
| **Development and external validation · 2020** | Denmark and Germany | We developed CLL-TIM using Danish patient records and validated it in an internal Danish test cohort and the external German CLL7 cohort. In the high-confidence internal benchmark, precision was 72% and recall 75%. [Nature Communications](https://doi.org/10.1038/s41467-019-14225-8) |
| **EHR deployment and validation · 2024** | Rigshospitalet and Roskilde, Denmark | We automated CLL-TIM predictions within an Epic-based EHR and evaluated performance in a later cohort, including patients with substantial missing data. [npj Digital Medicine](https://doi.org/10.1038/s41746-024-01132-6) |
| **PreVent-ACaLL: prospective validation and patient selection · 2026** | Denmark, Sweden and the Netherlands | We used CLL-TIM to select high-risk/high-confidence patients for pre-emptive treatment randomization and prospectively evaluated predictions in the non-randomized cohort. EHA 2026, S143 (**conference abstract**). [Results](docs/EVIDENCE.md#prevent-acall-prospective-validation-in-denmark-sweden-and-the-netherlands) · [NCT03868722](https://clinicaltrials.gov/study/NCT03868722) |
| **Independent US validation · 2026** | Rochester Epidemiology Project, USA | Independent investigators evaluated 454 patients; composite ROC-AUC was 0.74. AACR 2026, abstract 4228 (**conference abstract**). [Results](docs/EVIDENCE.md#independent-us-validation--aacr-2026) |

The [full validation results](docs/EVIDENCE.md) include cohort sizes, confidence selection, comparator models and uncertainty. PreVent-ACaLL evaluates whether short-term pre-emptive treatment can improve outcomes in patients selected by CLL-TIM.

## View a patient report

**[Open a complete CLL-TIM report](docs/synthetic_patient_examples/SynthPat01_Report.pdf)** showing the predicted two-year risk of infection and/or CLL treatment, predictive confidence and personalized risk factors.

[![CLL-TIM report for synthetic patient SynthPat01: risk of infection and/or CLL treatment, predictive confidence and personalized high- and low-risk factors.](docs/synthetic_patient_examples/SynthPat01.png)](docs/synthetic_patient_examples/SynthPat01_Report.pdf)

*Synthetic patient example. All nine example patients in this repository are synthetic. See the [technical notes on personalized risk factors](docs/IMPLEMENTATION.md#personalized-risk-factors) for the SHAP output conventions used by the scoring script.*

## Run CLL-TIM

From a terminal with Git and Conda available:

```sh
git clone https://github.com/RA19/cll-tim.git
cd cll-tim
conda env create -f environment-pipxgboost.yml
conda activate clltim_calc_env
python CLLTIM_MobileVer2.py
```

CLL-TIM reads the nine synthetic patients in `PatientFiles/SynthPat_Multi.csv` and writes their reports to `CLLTIMOutput/`. For an installation check that also compares results with the reference outputs, run:

```sh
python examples/verify_install.py
```

See the [usage guide](docs/USAGE.md) for expected files, environment requirements and troubleshooting.

## Prepare your patient data

CLL-TIM reads one CSV containing one or more patients, with one row per observation. The scoring script accepts baseline CLL markers, laboratory results and infection indicators. Follow the [input guide](docs/USAGE.md#patient-file-format), [data dictionary](PatientFileFormat.xlsx) and [complete example CSV](PatientFiles/SynthPat_Multi.csv).

Put your CSV in `PatientFiles/` and set `PatientFileName` near the top of `CLLTIM_MobileVer2.py`. Run CLL-TIM from the repository root. Reports are named using the patient identifier in the CSV.

## Further reading

| Guide | Contents |
|---|---|
| [Using CLL-TIM](docs/USAGE.md) | Installation, patient-file format, report interpretation and troubleshooting |
| [Validation and clinical use](docs/EVIDENCE.md) | Our retrospective and prospective validation, EHR deployment and independent US validation |
| [Model and methods](docs/METHODS.md) | Ensemble development, predictive confidence, missing data and personalized risk factors |
| [Technical reference](docs/IMPLEMENTATION.md) | Repository contents, model files, date lookup and SHAP conventions |
| [Figure sources](docs/figures/SOURCES.md) | Publication figures and attribution |

**Rudi Agius and collaborators** · Rigshospitalet / Copenhagen University Hospital · Technical University of Denmark. [Publications](docs/EVIDENCE.md#publications).
