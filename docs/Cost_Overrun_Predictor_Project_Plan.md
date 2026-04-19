# Infrastructure Project Risk Predictor

**Tagline:** An ML-based underwriting tool that predicts the probability of project distress and cost escalation on infrastructure mega-projects, designed for use by infra GPs and credit funds during pre-IC screening.

**Why this project for your profile:** It is the only project on a finance resume that simultaneously leverages (a) your structural engineering domain expertise, (b) your Stanford coursework in Renewable Project Finance and Financial Risk Analytics, (c) modern ML tooling, and (d) the actual workflow of infrastructure investors. It directly answers the recruiter's unspoken question: "Why is a structural engineer applying to infra PE?"

---

## The Pitch (one-paragraph version)

Infrastructure investment underwriting is fundamentally a question of distress risk: will this project reach operation, or will it be cancelled, renegotiated, or abandoned? The World Bank PPI database tracks 11,640 private infrastructure projects across 137 countries from 1990–2024, recording which ones became financially distressed or were cancelled. This project builds a gradient-boosted classifier that ingests project characteristics, macro context, and governance signals to output (1) probability of project distress and (2) for UK projects, expected cost baseline drift — with SHAP-based explainability so an analyst can defend the output in IC.

**Why distress prediction is the stronger framing for infra PE:** Academic cost-overrun research (Flyvbjerg et al.) measures percentage deviation from the original engineer's estimate. Infrastructure investors don't underwrite against engineer's estimates — they underwrite against the probability that a project fails to meet its investment case. Distress, cancellation, and renegotiation are the downstream events that trigger covenant breaches, equity impairments, and write-downs. Modeling these outcomes is directly analogous to default prediction in credit underwriting, which is a framework every infra PE associate immediately recognizes.

---

## Datasets

**Primary: World Bank Private Participation in Infrastructure (PPI) Database**
- Source: ppi.worldbank.org — public, free, no registration required
- 11,640 projects, 53 fields, covering energy, transport, water, ICT, and solid waste sectors
- Temporal coverage: 1990–2024, financial close year available
- Target: `status_n` — Active (10,800) / Distressed (260) / Cancelled (353) / Concluded (227)
  - Binary target: Distressed + Cancelled = 613 projects (5.3% base rate)
- Key features available at financial close: sector, subsector, project type (greenfield/brownfield/divestiture/management), country, region, income group, investment size (USD m), government support type, number of bidders, contract structure
- Downloaded: `data/raw/ppi_2024_full.csv`

**Secondary: UK Government Major Projects Portfolio (GMPP)**
- Source: gov.uk IPA annual reports — public, free, per-department CSVs
- Annual snapshots 2015–2024; ~200–300 projects per snapshot, 21 departments
- 68 of 227 projects in 2024 are "Infrastructure and Construction"
- Target: whole-life cost baseline drift across annual snapshots
  - Track each project's approved whole-life cost baseline from first appearance to most recent; baseline growth = cost escalation proxy
- Key features: project category, department, start/end dates, delivery confidence (RAG), annual financial year variance
- Downloaded: `data/raw/gmpp_2024_all_departments.csv`; historical 2015–2023 in `data/raw/gmpp_[year]_all_departments.csv`

**Feature Enrichment (Phase 2):**
- FRED: commodity price indices (steel, copper, cement), 10Y Treasury, CPI — matched to project financial close year
- World Bank Worldwide Governance Indicators: corruption, regulatory quality, rule of law — matched to country-year

---

## Target Variables

### PPI: Binary Distress Classifier

**Target:** `distressed_or_cancelled` = 1 if `status_n` in {Distressed, Cancelled}, else 0.

**Exposure window filter — FCY > 2017 projects are dropped entirely.**

Rationale: A project with FCY = 2021 that is still "Active" in the 2024 snapshot has had only 3 years of exposure. Labeling it as a negative case would be misleading — it simply hasn't had enough time to fail. Dropping FCY > 2017 ensures every project in the dataset has had at least 7 full years of exposure by the 2024 snapshot date. This eliminates survival-time confounding and aligns with typical infra PE hold periods (5–10 years). The specific 7-year window matches the question a credit committee answers when underwriting a 7-year senior secured infrastructure loan.

This decision is implemented in `src/data/preprocess_ppi.py`.

Pre-filter universe (all projects):

| Status | Count | % |
|---|---|---|
| Active | 10,800 | 92.8% |
| Concluded | 227 | 2.0% |
| Distressed | 260 | 2.2% |
| Cancelled | 353 | 3.0% |

Post-filter (FCY ≤ 2017 only): see `data/processed/ppi_model_ready.csv` for final counts.

**Two evaluation baselines — reported separately:**

1. **Calibration baseline** (for Brier score and calibration plots): constant prediction = observed distress rate in the training set (~6–8% post-filter). Measures whether the model is well-calibrated. A model that always outputs 7% beats this on Brier score only if it adds information.

2. **Random-ranking baseline** (for AUC): AUC = 0.50 by definition. AUC measures the model's ability to rank a distressed project above a non-distressed project — purely about discrimination, not calibration. Report AUC lift as: model AUC − 0.50.

Both baselines are reported in the evaluation section. They answer different questions: the calibration baseline asks "is the model correctly sized?"; the AUC baseline asks "can the model rank risk?"

### GMPP: Longitudinal Baseline Growth

**Target:** For projects appearing in multiple annual snapshots, compute:
`baseline_growth = (latest whole-life cost baseline - initial whole-life cost baseline) / initial whole-life cost baseline`

A project with 20% baseline growth has had its approved whole-life cost increase by 20% since first entry — the closest public equivalent to a cost overrun measurement for UK government infrastructure.

---

## Features

| Category | PPI fields | GMPP fields |
|---|---|---|
| Project | sector, subsector, type (greenfield/brown/divest), investment (log USD) | category (infra/ICT/transformation), baseline whole-life cost (£m) |
| Contract | contract type (BOT/BOOT/concession/etc.), number of bidders | n/a |
| Sponsor | public/private ratio, government support type | department |
| Macro | commodity price at FCY, 10Y Treasury at FCY, CPI | financial year |
| Governance | WGI scores (corruption, regulatory, rule of law) by country-FCY | RAG rating trajectory |
| Geography | region, income group, IDA eligible | n/a |
| Time | financial close year, project duration | snapshot year, project age |

---

## Modeling Approach

**Two-model architecture:**

1. **PPI Distress Classifier:** P(distress or cancellation) — Random Forest (primary, deployed), with XGBoost and LightGBM as secondary comparison models. Calibrated with Platt scaling (quantile-binned calibration curve).

   **Model selection rationale:** Random Forest was selected for deployment because it achieved superior cross-validated performance across all three metrics (AUC 0.925 vs XGBoost 0.915, Brier 0.035 vs 0.061, PR-AUC 0.710 vs 0.676). XGBoost and LightGBM are within 1–2 AUC points, providing cross-family robustness confirmation that the signal is real. `shap.TreeExplainer` supports `RandomForestClassifier` natively, so SHAP explainability works identically.

2. **GMPP Baseline Growth Regressor:** Expected % increase in whole-life cost baseline — OLS baseline → quantile regression (P50/P90) for tail risk. Secondary model for UK infrastructure projects with richer cost data.

**Validation — locked evaluation strategy:**

**Primary evaluation: 5-fold stratified cross-validation on the training set (FCY ≤ 2014).**
- Metrics reported: mean ± std of ROC-AUC, Brier score, and PR-AUC across 5 folds.
- Stratification on the target column to preserve class balance (7.2% positive rate) in each fold.
- All hyperparameter tuning (Optuna) is run inside cross-validation; no test-set information is used during tuning.
- Baselines reported at the same three metrics: constant-rate calibration baseline and logistic regression.

**Secondary: out-of-time test set (FCY 2015–2017), used as a ranking sanity check only.**
- The 2015–2017 cohort has a 0.1% distress rate in the 2024 PPI snapshot, which reflects PPI reporting lag — projects closed 2015–2017 have not yet had sufficient time to be re-labeled as distressed in the database, not a genuine signal that they are all healthy.
- Test-set AUC and rank-ordering are reported for completeness but are NOT the headline metric.
- Precision, recall, and F1 on the test set are not meaningful given the near-zero base rate and should not be cited in interviews.
- Interview-ready framing: *"True out-of-time evaluation was limited by reporting lag on recent cohorts — primary results are from 5-fold CV on the 8,435-project training set."*

**Imputation discipline:**
- All imputation statistics (medians, regional means) computed on the training set only, then applied to both train and test. Confirmed in `notebooks/features_script.py::compute_impute_params()`.
- One-hot encoding categories fit on train; test matrix reindexed to match train column list (fill_value=0 for unseen categories).

**Explainability:**
- SHAP values for global feature importance and per-project waterfall charts.
- Partial dependence plots for top 5 features.
- These are what make the model IC-defensible: "why did this project score high-risk?"

---

## Tech Stack

Python 3.11+ · pandas · numpy · scikit-learn · xgboost · lightgbm · shap · matplotlib · plotly · Streamlit (deployed on Streamlit Community Cloud) · GitHub.

---

## Deliverables

1. **GitHub repo** with clean notebooks (EDA → feature engineering → modeling → SHAP).
2. **Streamlit app** where a user enters project parameters (sector, type, country, investment size, year, governance score) and receives P(distress) + SHAP waterfall. *Highest-leverage deliverable for interviews.*
3. **One-page methodology PDF** for sharing with recruiters.
4. **Optional Medium / LinkedIn post** — "What I learned about infrastructure risk from 11,000 projects."

---

## Timeline

| Week | Milestone |
|---|---|
| 1 | Data acquisition (done), cleaning, feature engineering plan |
| 2 | EDA, PPI feature engineering, baseline models |
| 3 | XGBoost / LightGBM tuning, calibration, temporal validation |
| 4 | GMPP longitudinal tracker, SHAP analysis, write methodology |
| 5 | Streamlit app build + deploy |
| 6 | Polish, optional post, resume integration |

---

## Resume Bullets

**Primary bullet (Bullet C — combined story):**
> **Infrastructure Underwriting Risk Model (Python, XGBoost, SHAP, Deployed)** — Designed and deployed a two-layer risk model combining 11,640 World Bank PPI projects (binary distress classifier) with 10-year UK IPA longitudinal cost-escalation data; SHAP explainability for per-project IC defense; framed as probability of covenant-breach risk rather than academic cost-overrun percentage — the question infrastructure debt underwriters actually answer.

**Backup bullet (Bullet A — PPI scale):**
> **Infrastructure Project Distress Predictor (Python, XGBoost, SHAP)** — Trained a gradient-boosted classifier on 11,640 World Bank PPI infrastructure projects (1990–2024) to predict probability of project distress or cancellation within 7 years of financial close; validated with strict temporal split to eliminate lookahead bias; achieved [X]% AUC lift over random-ranking baseline.

**Note:** The GMPP longitudinal rigor angle (formerly Bullet B) is documented in the README as a standalone methodology section, not a separate resume bullet. The combined story (Bullet C) subsumes it.

---

## Interview Talking Points

- *"Why distress prediction instead of cost overrun %?"* — Cost overrun % requires planned vs. actual final cost, which isn't publicly available at scale. More importantly, distress is what investors actually underwrite: a GP doesn't lose money because a project was 15% over estimate — they lose money because the project was cancelled or couldn't service its debt. This model predicts the outcome that drives investment losses.
- *"Why XGBoost over a neural net?"* — Tabular data, ~11k samples, interpretability is a hard requirement for IC defensibility. Gradient boosting dominates on tabular data at this scale.
- *"Why temporal split?"* — Avoids lookahead bias and mimics real underwriting: you bet on future projects using only information available at the time of investment decision.
- *"What's the top driver of distress in your model?"* — Have a confident answer from SHAP. Likely candidates: project type (divestiture more stable than greenfield), sector (ICT highest distress rate), country governance scores, and government support structure.
- *"What are the limitations?"* — PPI only covers private-participation projects in emerging markets, so it misses OECD-country public infrastructure. The distress label is a coarser signal than actual cost overrun. Selection bias: projects that reached financial close are already past the initial screen.
- *"How would you extend this in production?"* — Layer in satellite/geospatial risk for geological complexity, ingest project-level news sentiment via LLM, and connect to a live commodity price feed for dynamic re-underwriting at each annual reporting date.
