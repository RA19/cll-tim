# Figure sources

[Overview](../../README.md) · [Evidence](../EVIDENCE.md) · [Methods](../METHODS.md)

## Publication figures

We reproduce figures from our model and deployment publications below. Consult the original captions for panel-specific populations, statistical tests and definitions.

**2020 source:** Agius R., et al. *Machine learning can identify newly diagnosed patients with CLL at high risk of infection.* Nature Communications 11, 363. [DOI and source](https://doi.org/10.1038/s41467-019-14225-8). © The Author(s) 2020, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

| File | Original figure | Scope |
|---|---|---|
| [nc_fig1_development.png](paper/nc_fig1_development.png) | Fig. 1 | Development workflow and intended clinical-trial selection |
| [nc_fig2_ensemble_composition.png](paper/nc_fig2_ensemble_composition.png) | Fig. 2 | Selected learners, variables and feature encodings |
| [nc_fig3_confidence_km.png](paper/nc_fig3_confidence_km.png) | Fig. 3 | Internal benchmark; BENCH-I n=288 with n=145 high-confidence predictions in panels a–c; internal test n=646 in panel d |
| [nc_fig4_external_missingness.jpg](paper/nc_fig4_external_missingness.jpg) | Fig. 4 | German external comparison; BENCH-E n=281 with n=142 T20–30-selected predictions for the benchmark panels; longitudinal missingness analysis |
| [nc_fig6_outcome_modelling.png](paper/nc_fig6_outcome_modelling.png) | Fig. 6 | Comparison of training outcomes (panel a, n=530 with full follow-up) and infection definitions (BENCH-I, panels b–c) |
| [nc_fig7_risk_factors.png](paper/nc_fig7_risk_factors.png) | Fig. 7 | Population factor analysis on n=3,720 Danish patients |
| [nc_fig9_multiple_paths.jpg](paper/nc_fig9_multiple_paths.jpg) | Fig. 9 | Historical web interface, patient-specific factors and patient heterogeneity |

**2024 source:** Agius R., et al. *Deployment and validation of the CLL treatment infection model adjoined to an EHR system.* npj Digital Medicine 7, 147. [DOI and source](https://doi.org/10.1038/s41746-024-01132-6). © The Author(s) 2024, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

| File | Original figure | Scope |
|---|---|---|
| [npj_fig6_missingness_layers.jpg](paper/npj_fig6_missingness_layers.jpg) | Fig. 6 | Conceptual feature-, algorithm- and ensemble-level missing-data handling |
| [npj_fig7_missingness_distributions.jpg](paper/npj_fig7_missingness_distributions.jpg) | Fig. 7 | Missing-feature distributions for correct/incorrect predictions on Bench-C n=1,099 |

These article licenses apply to the reproduced article material as specified in each publication; they do not establish a blanket license for the repository's code, weights or other media.

## Repository illustrations

| Asset | Provenance | Interpretation |
|---|---|---|
| [overview.svg](overview.svg) | CLL-TIM clinical overview | Clinical inputs, predicted two-year infection/treatment risk, predictive confidence and personalized risk factors |
| [fig_ensemble_votes.png](fig_ensemble_votes.png) and [ensemble_votes.csv](ensemble_votes.csv) | Generated from the nine synthetic feature vectors and trained ensemble by [clltim_showcase.py](../../examples/clltim_showcase.py) | Individual learner predictions and their means |
| [Synthetic reports](../synthetic_patient_examples/) | CLL-TIM reports for the bundled synthetic patients | Predicted infection/treatment risk, predictive confidence and personalized risk factors; [SHAP conventions](../IMPLEMENTATION.md#personalized-risk-factors) |
