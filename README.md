# Infrastructure Project Risk Predictor

An ML-based underwriting tool that predicts the probability of project distress and cost escalation on infrastructure mega-projects, designed for use by infrastructure GPs and credit funds during pre-IC screening.

---

## Overview

Infrastructure investment underwriting is fundamentally a question of distress risk: will this project reach operation, or will it be cancelled, renegotiated, or abandoned? This tool builds a Random Forest classifier on 11,640 World Bank PPI projects (1990–2024) to output P(project distress or cancellation within 7 years of financial close), with SHAP-based explainability for IC defensibility. XGBoost and LightGBM are included as cross-family robustness benchmarks.

Primary resume bullet:

> **Infrastructure Underwriting Risk Model (Python, Random Forest, SHAP, Deployed)** — Designed and deployed a two-layer risk model combining 11,640 World Bank PPI projects (binary distress classifier) with 10-year UK IPA longitudinal cost-escalation data; Random Forest selected over XGBoost/LightGBM on all three CV metrics (AUC 0.925, Brier 0.035, PR-AUC 0.710); SHAP explainability for per-project IC defense; framed as probability of covenant-breach risk rather than academic cost-overrun percentage.

---

## Setup

```bash
git clone https://github.com/<your-handle>/cost-overrun-predictor.git
cd cost-overrun-predictor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Data

**Primary:** World Bank Private Participation in Infrastructure (PPI) Database — 11,640 projects across 137 low/middle-income countries, 1990–2024. Free public download from ppi.worldbank.org.

**Secondary:** UK Government Major Projects Portfolio (GMPP) — 10 annual snapshots (2015–2024) from the Infrastructure and Projects Authority (IPA), downloaded per-department from gov.uk.

**Enrichment (Phase 2):** FRED commodity price indices and World Bank Worldwide Governance Indicators, matched to each project's financial close year and country.

### UK Infrastructure Cost Escalation (GMPP)

The GMPP longitudinal tracker follows UK government major projects across annual IPA snapshots to quantify whole-life cost baseline drift from initial approved budgets. Key findings from the 2015–2024 panel:

- 10 annual snapshots, 2,087 project-year observations, 21 departments
- "Infrastructure and Construction" category: 68 projects in the 2024 snapshot
- Baseline cost growth across snapshots serves as a cost escalation proxy for UK government infrastructure
- Leading indicators identified: delivery confidence (RAG) rating trajectory, project category, department

See `notebooks/` and `docs/PROJECT_PLAN.md` for full methodology.

---

## Methodology

**Target variable:** Binary distress-or-cancelled within 7 years of financial close. Projects with FCY > 2017 are excluded (insufficient exposure time). This aligns with typical infra PE hold periods (5–10 years) and eliminates survival-time confounding.

**Two-model architecture:**

1. **PPI Distress Classifier** — P(distress or cancellation within 7yr) using Random Forest with Platt scaling (primary deployed model). XGBoost and LightGBM included as cross-family benchmarks.
2. **GMPP Baseline Growth Regressor** — Expected % increase in UK project whole-life cost baseline. Secondary model.

**Validation:**
- Temporal split: train on FCY ≤ 2014 (8,435 projects), test on FCY 2015–2017 (1,048 projects, sanity check only — see reporting-lag caveat in `docs/PROJECT_PLAN.md`)
- Primary evaluation: 5-fold stratified CV on training set; metrics: AUC, Brier score, PR-AUC
- Calibration baseline: constant prediction at observed training distress rate (7.2%)
- Random-ranking baseline: AUC = 0.50

**Results (5-fold CV):**

| Model | AUC | Brier | PR-AUC |
|---|---|---|---|
| Constant baseline | 0.500 | 0.067 | 0.072 |
| Logistic Regression | 0.827 ± 0.010 | 0.161 ± 0.004 | 0.325 ± 0.012 |
| **Random Forest** | **0.925 ± 0.014** | **0.035 ± 0.002** | **0.710 ± 0.033** |
| XGBoost | 0.915 ± 0.014 | 0.061 ± 0.002 | 0.676 ± 0.021 |
| LightGBM | 0.906 ± 0.015 | 0.071 ± 0.002 | 0.663 ± 0.033 |

All predictions paired with SHAP waterfall explanations.

---

## Results

*To be populated after model training (Day 3–4).*

Key metrics: AUC-ROC, precision-recall AUC, Brier score; lift vs. naive base rate and random baselines.

---

## Demo

*Streamlit app link to be added after deployment (Day 5).*

To run locally:

```bash
streamlit run src/app/app.py
```

---

## Repository Structure

```
cost-overrun-predictor/
├── data/           # raw, interim, processed (not committed)
├── notebooks/      # EDA → features → modeling → SHAP
├── src/            # data ingestion, features, models, app
├── models/         # serialized model artifacts
├── tests/          # pytest suite for data pipeline
└── docs/           # project plan, session log, methodology PDF
```

---

## Future Work: Developed-Market Extension

Version 1 deliberately focuses on emerging-market private infrastructure (World Bank PPI) and UK government major projects (GMPP) — a deliberate scope decision to ensure defensible modeling depth within a constrained timeline rather than shallow coverage of many markets. The natural next layer is a developed-market distress and cost-escalation model covering OECD infrastructure.

**Data sources to add:**

- **US FHWA Major Projects Database** (highways.dot.gov/federal-lands/projects) — federal highway mega-projects with committed cost, schedule, and variance data; ~200 active projects with multi-year tracking
- **US DOT Build America Bureau project pipeline** (buildamerica.dot.gov/projects) — federal credit-assisted infrastructure (TIFIA/RRIF loans), includes project cost, sector, and sponsor type; ~180 projects since 2000
- **European TEN-T Project Portal** (tentec.ec.europa.eu) — major EU trans-European transport network projects across 27 member states; cost, delivery stage, and co-funding data; ~500 projects
- **GMPP historical snapshots** — additional pre-2015 IPA data (available via FOI or archived gov.uk releases) to extend the UK panel beyond the 10 years already ingested
- **Country-level sources:** Japan MLIT major project tracker, Australia Infrastructure Australia National Priority List, Canada Infrastructure Bank project registry — each adds ~50–150 projects with cost and schedule baselines

**Architectural decision:**

A developed-market extension would be a separate model, not additional rows in the existing PPI model. The data-generating process is fundamentally different: public-sector sponsors, regulated utility structures, stronger rule-of-law environments, and EU procurement rules vs. the PPI universe of private concessions in low/middle-income countries. Appending rows would conflate these regimes and degrade both models. The intended architecture is two independently trained classifiers sharing a common 87-feature schema where possible (WGI scores, macro environment at financial close, project type, investment size), deployed behind the same Streamlit interface with a **region mode toggle** (Emerging Markets / OECD) that routes inputs to the appropriate model and renders the corresponding SHAP waterfall.

**Estimated effort:** 40–60 hours, primarily data harmonization — standardizing cost-baseline definitions across FHWA, TEN-T, and GMPP schemas — plus retraining and recalibrating the classifier on the new label distribution.

---

## Author

Nicholas Daal — Stanford MS Structural Engineering. Background: structural engineering on $1B+ infrastructure mega-projects; finance training at Eastdil Secured.
