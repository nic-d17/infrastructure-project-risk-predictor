# Session Log

---

## Day 1 — 2026-04-18

**Goal:** Environment setup + data acquisition decision.

**Shipped:**
- Full folder structure created per CLAUDE.md spec
- `requirements.txt` with pinned dependencies
- `.gitignore` for Python ML project
- `README.md` skeleton
- `docs/SESSION_LOG.md` initialized

**Data decision pending:** Awaiting selection of Flyvbjerg dataset acquisition path (3 options proposed). No data downloaded yet.

**Blocked:** None.

**Data acquired:**
- `data/raw/ppi_2024_full.dta` + `ppi_2024_full.csv` — World Bank PPI, 11,640 projects
- `data/raw/ppi_2024_decoded.csv` — full STATA value labels decoded
- `data/interim/ppi_data_dictionary.txt` — field-by-field value label map
- `data/raw/gmpp_2024_all_departments.csv` — UK GMPP 2024, 227 rows, 21 departments
- `data/raw/gmpp_{2015-2023}_all_departments.csv` — 9 historical snapshots
- `data/raw/gmpp_all_years.csv` — longitudinal stack, 2,087 rows, 10 years

**Target variables confirmed:**
- PPI: `status_n` → binary distress-or-cancelled (5.3% base rate, 613 positive cases)
- GMPP: whole-life cost baseline drift across annual snapshots

**Tomorrow (Day 2):** Feature engineering (`02_features.ipynb`) based on EDA hypotheses.

---

## Day 2 — 2026-04-19

**Shipped:**
- `src/data/preprocess_ppi.py` — 7-year exposure filter (FCY ≤ 2017), temporal split, model-ready CSV
- `data/processed/ppi_model_ready.csv` — 9,483 rows × 26 cols, 6.4% distress rate
- `notebooks/01_eda.ipynb` — executed, all 7 EDA sections complete
- `notebooks/figures/` — 8 charts generated
- Docs updated: PROJECT_PLAN.md (7-year target, bullet hierarchy, dual baselines), README.md, CLAUDE.md

**Key EDA findings:**
- Pre-2000 cohort: 13.4% distress; 2015-2017: ~0.1% (possible PPI reporting lag — flag for model diagnostics)
- ICT highest-risk sector (9.9%); MSW 0% (small n, stable utility contracts)
- CAM finding inverts standard thesis: competitive bidding = 7% distress vs. direct negotiation 2.2% (confounded by project complexity, not procurement quality)
- LAC + AFR: >10% distress, 3-4× lower-risk regions
- No multicollinearity in numeric features; investment log-normal ✓

**Top 3 hypotheses logged:** sector signal, CAM-complexity proxy, governance amplifier

**Blocked:** None.

---

## Day 3 — 2026-04-19 (continued)

**Goal:** Feature engineering corrections + baseline modeling.

**Shipped:**
- `notebooks/features_script.py` — 3 corrections: (1) imputation leakage fixed (stats computed on train FCY≤2014 only, applied to test); (2) interaction terms pruned from 69 one-hot dummies to 3 label-encoded integers; (3) OHE categories fit on train, test reindexed to match
- `data/processed/features_train.parquet` — 8,435 × 87 features (7.2% distress)
- `data/processed/features_test.parquet` — 1,048 × 87 features (0.1% distress, reporting-lag artefact)
- `notebooks/02_features.ipynb` — re-executed with corrected pipeline
- `docs/PROJECT_PLAN.md` — locked evaluation strategy with reporting-lag caveat
- `notebooks/03_modeling.ipynb` — executed, all 7 sections complete
- `models/` — calibrated + uncalibrated XGBoost, best_params.json, feature_cols.json

**Modeling results (5-fold stratified CV, train set):**

| Model | AUC | Brier | PR-AUC |
|---|---|---|---|
| Constant baseline | 0.500 | 0.067 | 0.072 |
| Logistic Regression | 0.827 ± 0.010 | 0.161 ± 0.004 | 0.325 ± 0.012 |
| Random Forest | 0.925 ± 0.014 | 0.035 ± 0.002 | 0.710 ± 0.033 |
| XGBoost (tuned) | 0.915 ± 0.014 | 0.061 ± 0.002 | 0.676 ± 0.021 |
| LightGBM (tuned) | 0.906 ± 0.015 | 0.071 ± 0.002 | 0.663 ± 0.033 |

Winner: XGBoost (AUC +0.415 over random, per-fold range 0.893–0.932).

**Tomorrow (Day 4):** `04_shap.ipynb` — global importance, per-project waterfall, partial dependence plots.
