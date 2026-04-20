# Infrastructure Project Risk Predictor

An ML-based underwriting tool that estimates the probability of project distress for infrastructure mega-projects, designed for pre-IC screening by infrastructure GPs and credit funds.

**Live demo:** *(Streamlit Cloud URL — add after deployment)*

---

## What it does

Infrastructure debt underwriting is fundamentally a distress risk question: will this project reach operation, or will it be cancelled, renegotiated, or abandoned? This tool outputs **P(distress or cancellation within 7 years of financial close)** with SHAP-based explainability for IC-defensible scoring.

Trained on 11,640 World Bank PPI projects across 137 low/middle-income countries (1990–2024). At financial close, the model consumes project structure, governance, macro environment, and contract terms — the same inputs an underwriter assembles at deal screening.

---

## Model performance (5-fold stratified CV)

| Model | AUC | Brier | PR-AUC |
|---|---|---|---|
| Constant baseline | 0.500 | 0.067 | 0.072 |
| Logistic Regression | 0.827 ± 0.010 | 0.161 ± 0.004 | 0.325 ± 0.012 |
| **Random Forest** | **0.925 ± 0.014** | **0.035 ± 0.002** | **0.710 ± 0.033** |
| XGBoost | 0.915 ± 0.014 | 0.061 ± 0.002 | 0.676 ± 0.021 |
| LightGBM | 0.906 ± 0.015 | 0.071 ± 0.002 | 0.663 ± 0.033 |

Random Forest selected for deployment: wins on all three metrics, gaps exceed CV noise, and `shap.TreeExplainer` supports it natively.

---

## Key findings (SHAP analysis)

| Category | Share of predictive signal |
|---|---|
| Governance (WGI) | 23.9% |
| Project Type / Structure | 18.1% |
| Macro at financial close | 16.0% |
| Contract / Procurement | 12.8% |
| Size / Investment | 8.4% |
| Sector | 7.6% |
| Geography | 6.8% |
| Cohort (temporal) | 6.3% |

Top features: `log_investment`, `treasury_10y`, `energy_idx`, `wgi_regulatory_quality`, `metals_idx`. Leakage audit confirmed all top-20 features are observable at financial close year.

**Hypothesis outcomes:**
- H1 (sector dominance): Weak — sector risk is distributed across interaction terms, not raw sector dummies
- H2 (CAM as complexity proxy): Supported — competitive bidding projects show 7% distress vs 2.2% for direct negotiation; procurement bucket = 12.8% of signal
- H3 (WGI sharpens geography): Strongly supported — WGI = largest single category; `wgi_regulatory_quality` ranks #4 overall

---

## Methodology

**Target variable:** Binary distress-or-cancelled within 7 years of financial close (613 positive cases, 7.2% rate). Projects with FCY > 2017 excluded — insufficient exposure time creates survival-time confounding. This aligns with typical infra PE hold periods (5–10 years) and is the question a credit committee is actually answering when underwriting infrastructure debt.

**Temporal validation:** Train on FCY ≤ 2014 (8,435 projects), sanity-check on FCY 2015–2017 (1,048 projects). The test-set near-zero distress rate (0.1%) reflects PPI reporting lag — not genuine signal — so the primary evaluation metric is 5-fold stratified CV on the training set.

**Feature engineering (87 features):**
- WGI governance scores joined by country × financial-close-year (clamped to 1996 earliest)
- Macro environment: US 10Y Treasury, World Bank energy and metals indices at FCY
- Three label-encoded interaction terms (CAM × sector, sector × type, region × cohort) replacing 69 one-hot dummies — reduces overfitting risk with 608 positive cases
- Train-only imputation discipline: all medians, regional means, and label maps computed on FCY ≤ 2014 only, then applied to both splits

**Calibration:** Platt scaling (`CalibratedClassifierCV`, sigmoid, 5-fold), quantile-binned calibration curve.

---

## Streamlit app

The deployed app takes deal-time inputs (sector, country, project type, contract structure, investment size) and outputs:
- Calibrated P(distress) with color-coded risk bucket (Low / Moderate / Elevated / High)
- SHAP waterfall showing the top 10 drivers for this specific prediction
- Category-level breakdown of predictive signal

WGI governance scores and macro values are auto-populated from country and financial close year. Three example projects (high-risk / low-risk / decision boundary) load in one click for live demos.

To run locally:
```bash
git clone https://github.com/nicholasdaal/infrastructure-project-risk-predictor.git
cd infrastructure-project-risk-predictor
python3 -m venv venv && source venv/bin/activate
pip install -r requirements_streamlit.txt
streamlit run src/app/streamlit_app.py
```

---

## Repository structure

```
infrastructure-project-risk-predictor/
├── notebooks/
│   ├── 01_eda.ipynb            # EDA: distress rates by sector, region, cohort
│   ├── 02_features.ipynb       # Feature engineering + leakage audit
│   ├── 03_modeling.ipynb       # Model comparison, calibration, fold stability
│   └── 04_shap.ipynb           # SHAP: global importance, hypotheses, waterfalls
├── src/app/
│   ├── streamlit_app.py        # Streamlit UI
│   └── predictor.py            # Encoding pipeline + prediction + SHAP figures
├── models/
│   ├── best_model_calibrated.pkl
│   ├── best_model_uncalibrated.pkl
│   ├── feature_cols.json
│   ├── impute_params.pkl       # Train-only imputation stats
│   ├── country_lookup.csv
│   ├── wgi_by_country.csv
│   └── macro_lookup.csv
├── data/                       # raw/, interim/, processed/ (raw not committed)
├── docs/
│   ├── Cost_Overrun_Predictor_Project_Plan.md
│   └── SESSION_LOG.md
├── requirements.txt            # Full research environment
└── requirements_streamlit.txt  # Minimal deps for Streamlit Cloud
```

---

## Data sources

**Primary:** [World Bank PPI Database](https://ppi.worldbank.org) — 11,640 projects, 137 countries, 1990–2024. Free public download.

**Enrichment:** [World Bank WGI](https://info.worldbank.org/governance/wgi/) — country-year governance scores (6 indicators, 1996–2023). [World Bank Pink Sheet](https://www.worldbank.org/en/research/commodity-markets) — monthly commodity price indices. [FRED GS10](https://fred.stlouisfed.org/series/GS10) — US 10Y Treasury annual averages.

**Secondary (longitudinal cost escalation):** UK IPA Government Major Projects Portfolio (GMPP) — 10 annual snapshots (2015–2024), downloaded from [gov.uk](https://www.gov.uk/government/collections/major-projects-data).

---

## Future work: Developed-market extension

Version 1 deliberately focuses on emerging-market private infrastructure (World Bank PPI) and UK government major projects (GMPP) — a deliberate scope decision to ensure defensible modeling depth within a constrained timeline rather than shallow coverage of many markets. The natural next layer is a developed-market distress and cost-escalation model covering OECD infrastructure.

**Data sources to add:**

- **US FHWA Major Projects Database** (highways.dot.gov/federal-lands/projects) — federal highway mega-projects with committed cost, schedule, and variance data; ~200 active projects with multi-year tracking
- **US DOT Build America Bureau project pipeline** (buildamerica.dot.gov/projects) — federal credit-assisted infrastructure (TIFIA/RRIF loans), includes project cost, sector, and sponsor type; ~180 projects since 2000
- **European TEN-T Project Portal** (tentec.ec.europa.eu) — major EU trans-European transport network projects across 27 member states; cost, delivery stage, and co-funding data; ~500 projects
- **GMPP historical snapshots** — additional pre-2015 IPA data (available via FOI or archived gov.uk releases) to extend the UK panel beyond the 10 years already ingested
- **Country-level sources:** Japan MLIT major project tracker, Australia Infrastructure Australia National Priority List, Canada Infrastructure Bank project registry — each adds ~50–150 projects with cost and schedule baselines

**Architectural decision:** A developed-market extension would be a separate model, not additional rows in the existing PPI model. The data-generating process is fundamentally different: public-sector sponsors, regulated utility structures, stronger rule-of-law environments, and EU procurement rules vs. the PPI universe of private concessions in low/middle-income countries. Appending rows would conflate these regimes and degrade both models. The intended architecture is two independently trained classifiers sharing a common 87-feature schema where possible, deployed behind the same Streamlit interface with a **region mode toggle** (Emerging Markets / OECD).

**Estimated effort:** 40–60 hours, primarily data harmonization across FHWA, TEN-T, and GMPP schemas.

---

## Author

Nicholas Daal — Stanford MS Structural Engineering (June 2026). Background: structural engineering on $1B+ infrastructure mega-projects (TYLin, Kiewit, EllisDon); finance training at Eastdil Secured; CFA Level I candidate.

Resume bullet:
> **Infrastructure Underwriting Risk Model** (Python · Random Forest · SHAP · Streamlit) — Built and deployed a distress classifier on 11,640 World Bank PPI infrastructure projects; RF selected over XGBoost/LightGBM across all three CV metrics (AUC 0.925, Brier 0.035, PR-AUC 0.710); SHAP leakage audit; per-project IC-defensible waterfall explanations; framed as probability of covenant-breach risk rather than academic cost-overrun percentage.
