# Validation and clinical use

[Overview](../README.md) · [Using CLL-TIM](USAGE.md) · [Methods](METHODS.md) · [Technical reference](IMPLEMENTATION.md)

We developed CLL-TIM in Denmark, validated its predictions in Denmark and Germany, and extended prospective validation through PreVent-ACaLL in Denmark, Sweden and the Netherlands. We also integrated CLL-TIM into the Danish EHR. Independent US validation provides a further assessment of generalization beyond the European cohorts.

## Clinical endpoint and prediction point

We modeled the composite outcome of **infection and/or initiation of CLL treatment within two years of the prediction point**. For development, we set the prediction point at three months after diagnosis to include investigations performed during the diagnostic work-up. Patients who had already initiated treatment or died before that point were excluded. Timing in the internal and external test cohorts is detailed in our [2020 publication and supplement](#publications).

We used a blood-culture draw as a proxy for infection, irrespective of whether the culture was positive, negative or contaminated. In prospective PreVent-ACaLL follow-up, infection is defined as CTCAE grade ≥3. These definitions should be considered when comparing outcomes across cohorts.

## Development and external validation · 2020

We identified 4,149 patients in the Danish National CLL Registry. After prediction-point exclusions, 3,720 patients remained: 2,432 for training, 642 for validation and 646 for the held-out internal test cohort.

For comparison with CLL-IPI, our internal benchmark (BENCH-I) included 288 patients with complete CLL-IPI variables and two-year follow-up. CLL-TIM made 145 high-confidence predictions in this subset, achieving **72% precision, 75% recall and an MCC of approximately 0.56**. These metrics describe the high-confidence benchmark subset.

We externally validated CLL-TIM in the German CLL7 cohort. The benchmark comparison (BENCH-E) included 281 patients with complete CLL-IPI and two-year follow-up. We selected 142 predictions using T20–30: the top 20% predicted high risk and bottom 30% predicted low risk. CLL-TIM separated high- and low-risk patients despite substantial missing longitudinal features. Two-year event-free survival was 43.9% and 75.3% in these selected high- and low-risk groups, respectively.

[Nature Communications, 2020](https://doi.org/10.1038/s41467-019-14225-8) · [PDF](papers/Agius2020_NatCommun_CLLTIM.pdf) · [Supplement](papers/Agius2020_NatCommun_CLLTIM_Supplementary.pdf)

### Internal Danish validation

![CLL-TIM internal Danish validation: infection- and treatment-free survival, cumulative incidence, CLL-IPI comparison and missing-feature rates.](figures/paper/nc_fig3_confidence_km.png)

*Our 2020 Figure 3. Panels a–c: BENCH-I, n=288 with n=145 high-confidence predictions. Panel d: internal test cohort, n=646. Curves and comparisons retain the uncertainty and selection rules shown in the original figure.*

### External German validation

![External German validation of CLL-TIM under substantial longitudinal missingness.](figures/paper/nc_fig4_external_missingness.jpg)

*Our 2020 Figure 4. BENCH-E comparison: n=281, including n=142 T20–30-selected predictions. CLL-IPI variables were complete in the comparison subset; substantial missingness concerned other longitudinal CLL-TIM features.*

## EHR deployment and validation · 2024

We integrated CLL-TIM into an Epic-based Danish EHR, automating data retrieval, harmonization, prediction and presentation of the results in the patient record. Physicians receive the predicted risk of infection and/or CLL treatment, predictive confidence and personalized risk factors within the EHR.

We evaluated CLL-TIM using patients from **Rigshospitalet and Roskilde** diagnosed from 2018 onwards. The population-based deployment cohort contained 1,480 patients; the analyzed deployment benchmark contained 1,099. We encountered substantially more missing structured baseline data than in the research EHR, including Binet stage and IGHV status.

On Bench-B (n=1,099), approximately 25% of predictions were high confidence. We found an MCC of 0.500 for high-confidence predictions, compared with 0.289 when including all predictions. CLL-IPI could be calculated for only 7/1,099 patients under the deployment missing-data conditions. We also explored a high-risk/high-confidence threshold of 0.65; prospective assessment on a new patient set is required for this adjustment. The scoring script uses 0.58.

Our EHR integration includes automated monitoring of predictions and input data. Predictions were labeled research-use only while clinical-trial evaluation continued.

[npj Digital Medicine, 2024](https://doi.org/10.1038/s41746-024-01132-6) · [Open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11153589/) · [PDF](papers/Agius2024_npjDigitMed_CLLTIM_deployment.pdf)

### Missing data and prediction performance

![Missing-feature distributions in patients with correct and incorrect CLL-TIM predictions.](figures/paper/npj_fig7_missingness_distributions.jpg)

*Our 2024 Figure 7; Bench-C, n=1,099. We found similar missing-feature distributions for true versus false positives and true versus false negatives. We also identified sensitivity to missing key variables, including Binet stage.*

## PreVent-ACaLL: prospective validation in Denmark, Sweden and the Netherlands

We used CLL-TIM in PreVent-ACaLL to identify newly diagnosed patients at high risk of infection and/or early CLL treatment who did not yet meet iwCLL treatment criteria. High-risk/high-confidence patients were eligible for randomization between observation and three months of pre-emptive acalabrutinib plus venetoclax. Follow-up also enabled prospective validation of CLL-TIM across Denmark, Sweden and the Netherlands.

**EHA 2026, S143 (conference abstract).** We assessed 574 patients across nine sites. Of 95 high-risk/high-confidence patients, 40 entered the randomized component; the reported non-randomized outcomes concern the remaining 55. Their two-year infection-/treatment-free survival was 54%, compared with 74.7% for high-risk/low-confidence, 90.3% for low-risk/low-confidence and 95.9% for low-risk/high-confidence patients. Infection was defined as CTCAE grade ≥3. These results validate risk stratification and predictive confidence in the non-randomized cohort; the randomized comparison addresses treatment benefit.

[EHA abstract S143](https://library.ehaweb.org/eha/2026/eha-2026/4206697/carsten.utoft.niemann.prospective.validation.of.the.cll-tim.machine-learning.html) · [PreVent-ACaLL](https://chip.dk/research-areas/clinical-trials/studies-and-projects/prevent-acall/) · [NCT03868722](https://clinicaltrials.gov/study/NCT03868722)

## Independent US validation · AACR 2026

**AACR 2026, abstract 4228 (conference abstract).** Independent investigators evaluated CLL-TIM in 454 newly diagnosed patients from the Rochester Epidemiology Project. Composite ROC-AUC was 0.74 (95% CI 0.68–0.80), compared with 0.69 for CLL-IPI (p=0.11) and 0.68 for IPS-E (p=0.02). Infection-only AUC was 0.68. Most discrimination was attributable to treatment prediction.

[Abstract DOI](https://doi.org/10.1158/1538-7445.am2026-4228) · [Official AACR proceedings, abstract 4228](https://www.aacr.org/wp-content/uploads/2026/04/AACR2026_Proceedings_Part-1.pdf)

## Publications

- **Model:** Agius R., et al. *Machine learning can identify newly diagnosed patients with CLL at high risk of infection.* Nature Communications 11, 363 (2020). [DOI](https://doi.org/10.1038/s41467-019-14225-8).
- **Deployment:** Agius R., et al. *Deployment and validation of the CLL treatment infection model adjoined to an EHR system.* npj Digital Medicine 7, 147 (2024). [DOI](https://doi.org/10.1038/s41746-024-01132-6).
- **Recommendations:** Agius R., et al. *Artificial intelligence models in chronic lymphocytic leukemia—recommendations toward state-of-the-art.* Leukemia & Lymphoma 63, 265–278 (2022). [DOI](https://doi.org/10.1080/10428194.2021.1973672).
- **Prospective validation:** PreVent-ACaLL non-randomized cohort, EHA Congress 2026, S143, linked above.
- **US validation:** Mwangi R., et al. Cancer Research 86(7 Supplement), abstract 4228 (AACR 2026), linked above.

Publication figures: © the original authors, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [Figure sources and attribution](figures/SOURCES.md).
