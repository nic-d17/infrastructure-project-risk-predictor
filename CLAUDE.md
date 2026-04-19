# CLAUDE.md — Project Context

This file gives you (Claude Code) persistent context for this project. Read it at the start of every session.

## Project: Infrastructure Project Risk Predictor

An ML-based underwriting tool that predicts the probability of project distress and cost escalation on infrastructure mega-projects. Built as a portfolio project to support recruiting for infrastructure investment banking and infrastructure private equity roles (Macquarie, Brookfield, GIP, Stonepeak, Lazard P&R, etc.).

The full project plan lives at `docs/PROJECT_PLAN.md`. Read it before starting any new task.

## Owner

Nicholas Daal — Stanford MS Structural Engineering (Jun 2026), UC Berkeley BS Civil Engineering. Background: structural engineering on $1B+ infrastructure mega-projects (TYLin, Kiewit, EllisDon). Finance training: Eastdil Secured, CFA Level I candidate. Coursework in Renewable Project Finance, Investment Science, Financial Risk Analytics.

## Goals

1. Ship a working ML model with proper temporal validation by end of Day 4.
2. Ship a deployed Streamlit app with live demo URL by end of Day 5.
3. Ship a public GitHub repo with clean README and one-page methodology PDF by end of Day 6.
4. Output: defensible bullets for the resume + interview talking points.

## Tech Stack (do not deviate without asking)

- Python 3.11+, managed with `venv`
- pandas, numpy, scikit-learn for data + baselines
- xgboost, lightgbm for primary models
- shap for explainability
- streamlit for the demo app, deployed on Streamlit Community Cloud
- matplotlib + plotly for visualization
- pytest for tests on data pipeline
- git + GitHub for version control

## Folder Structure (target)

```
cost-overrun-predictor/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── docs/
│   └── PROJECT_PLAN.md
├── data/
│   ├── raw/           # untouched downloads
│   ├── interim/       # intermediate joins
│   └── processed/     # final modeling-ready tables
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_shap.ipynb
├── src/
│   ├── data/          # ingestion + cleaning scripts
│   ├── features/      # feature engineering
│   ├── models/        # train, evaluate, predict
│   └── app/           # streamlit app code
├── models/            # serialized model artifacts (.pkl)
└── tests/
```

## Data Sources

**Primary: World Bank Private Participation in Infrastructure (PPI) Database**
- 11,640 projects across 137 low/middle-income countries, 1990–2024
- Downloaded as .dta, converted to CSV: `data/raw/ppi_2024_full.csv`
- Target variable: `status_n` — binary distress-or-cancelled (613 projects, 5.3%) vs. active-or-concluded
- Key fields: sector, project type (greenfield/brownfield/divestiture), country, region, income group, financial close year, investment (USD m), government support, contract type

**Secondary: UK Government Major Projects Portfolio (GMPP)**
- Annual snapshots 2015–2024 from Infrastructure and Projects Authority (IPA)
- 21 UK government departments; ~200–300 projects per year
- 68 "Infrastructure and Construction" projects in 2024 snapshot
- Target variable: baseline cost growth across annual snapshots (whole life cost baseline drift = cost overrun proxy)
- Key fields: project name, department, annual report category, financial year baseline (£m), whole life cost baseline (£m), annual variance (%), delivery confidence (RAG rating)

**Enrichment (Phase 2):**
- FRED API: commodity price indices (steel, copper, cement) and 10Y Treasury by year
- World Bank Worldwide Governance Indicators: country-year governance scores

## Target Variables — Critical Context

**Why distress/cancellation is the right PPI target:**
The PPI database does not record planned vs. actual final construction cost. It records investment commitments at financial close and project operational status. Project distress and cancellation are the downstream consequence of cost overruns, schedule failures, and revenue shortfalls — they are the outcome that infra PE GPs and credit funds are actually underwriting against. A project that becomes distressed has, by definition, experienced a failure of the underwriting assumptions. This is more directly actionable than an academic overrun percentage.

**Why GMPP baseline growth is the right secondary target:**
The UK IPA publishes whole-life cost baselines annually. Tracking how a project's baseline drifts upward from its initial approved budget across annual snapshots gives a longitudinal cost escalation signal. This is the closest available public proxy to "cost overrun at reporting date" for UK government major projects.

**These are practitioner-relevant risk proxies, not academic overrun measurements.** In an infra PE interview, frame this as: "I modeled the probability a project enters financial distress — which is the question a credit committee is actually answering when they underwrite infrastructure debt."

## Critical Rules

1. **Never fabricate data.** If a dataset isn't accessible, surface it immediately and propose alternatives. Do not make up rows or values to fill gaps.
2. **Use temporal splits, not random splits, for validation.** Train on projects with financial close before 2015, test on 2015–2024. This is non-negotiable — it mimics real underwriting and avoids lookahead bias.
3. **Always compare against a naive baseline** (e.g., "always predict 5.3% distress rate" and a logistic regression baseline). The lift over baseline is the headline metric.
4. **Interpretability is a hard requirement.** Every model output in the Streamlit app must be paired with a SHAP explanation. No black-box predictions.
5. **Ask before installing new top-level dependencies.** Keep the stack lean.
6. **Commit at logical checkpoints.** After each notebook completes, after each src module is tested. Use conventional commit prefixes (feat, fix, docs, test, refactor).
7. **No emojis in code, commits, or the README** unless explicitly asked.
8. **The Flyvbjerg database is off the table.** Do not suggest emailing Flyvbjerg or pursuing the Oxford Global Projects database. The project proceeds entirely on PPI + GMPP.

## Session Workflow

At the start of each session: read this file, then read `docs/PROJECT_PLAN.md`, then check `git log --oneline -10` to see where we left off. Confirm the current day's milestone before starting work.

At the end of each session: commit progress, write a short note in `docs/SESSION_LOG.md` summarizing what shipped and what's blocked, and stage tomorrow's first task.

## Interview Lens (keep in mind during all decisions)

Every choice should be defensible to an infra PE associate in a 30-minute case study. Favor simple, interpretable, well-validated models over fancy architectures. The story matters as much as the AUC.

The framing is: **"I built a project distress predictor — the question an IC is actually answering."** Not: "I tried to replicate Flyvbjerg."
