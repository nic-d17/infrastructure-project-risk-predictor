"""Build notebooks/02_features.ipynb from features_script.py logic."""
import nbformat as nbf
from pathlib import Path

NB_PATH = Path(__file__).parent / "02_features.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

cells = []
def md(text): return nbf.v4.new_markdown_cell(text)
def code(src): return nbf.v4.new_code_cell(src)


# ── Title ─────────────────────────────────────────────────────────────────
cells.append(md("""# 02 — Feature Engineering: PPI Distress Classifier
**Infrastructure Project Risk Predictor**

Pipeline:
1. Information-at-FCY audit — classify every column as usable / forbidden / excluded
2. Load 9,483 projects (FCY ≤ 2017, post-exposure-filter)
3. Enrichment: WGI governance scores (country × year) + Pink Sheet commodity prices + US 10Y Treasury
4. Feature encoding: numeric transforms, one-hot categoricals, binary flags
5. Interaction terms: CAM × sector, sector × type, region × fcy_cohort
6. Output: `features_train.parquet` (FCY ≤ 2014) and `features_test.parquet` (FCY 2015–2017)

**Key discipline:** Every feature must be observable at financial close (FCY). No post-FCY outcome fields.
"""))


# ── Setup ──────────────────────────────────────────────────────────────────
cells.append(md("## Setup"))
cells.append(code("""\
import warnings; warnings.filterwarnings("ignore")
import sys, os, importlib.util
from pathlib import Path

# ROOT is one level up from notebooks/
ROOT = Path(os.getcwd())
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

NOTEBOOKS_DIR = ROOT / "notebooks"
sys.path.insert(0, str(NOTEBOOKS_DIR))

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 60)

# Import features_script as a module (avoids __file__ issues with exec())
spec = importlib.util.spec_from_file_location(
    "features_script", NOTEBOOKS_DIR / "features_script.py"
)
fs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs)

print("features_script loaded OK")
print(f"ROOT: {ROOT}")
"""))


# ── Section 1: Audit ──────────────────────────────────────────────────────
cells.append(md("""## 1 — Information-at-FCY Audit

Every PPI column classified: *"Could an investor observe this at financial close?"*

- **USABLE** — observable at FCY, safe to use as a model feature
- **FORBIDDEN** — encodes post-FCY outcome; using it = data leakage
- **EXCLUDED** — >50% missing or confounded construction
- **ADMIN** — identifier / split label, not a feature
- **ENRICHMENT** — join key only, dropped before modeling
"""))

cells.append(code("""\
fs.print_audit()
"""))

cells.append(md("""\
**Forbidden note:** `status_n` and `target` are the outcome labels. Using them as features would be perfect leakage — the model would simply memorize the answer.

**Excluded rationale:**
- `numberb` (>88% missing) — imputing would fabricate the competitive-bidding signal for most of the dataset
- `pcapacity` — requires final installed capacity, which may not be known at FCY for greenfield projects
- `technol` (61% missing) — retained signal is captured via sector/subsector; missingness exceeds 50% threshold
- `PRS` — a data quality score assigned by PPI researchers after the fact; not observable by an investor
"""))


# ── Section 2: Load & Describe ─────────────────────────────────────────────
cells.append(md("## 2 — Load Preprocessed Data"))

cells.append(code("""\
df = fs.load_data()
print(f"\\nFCY range: {df['FCY'].min():.0f}–{df['FCY'].max():.0f}")
df[["FCY","target","sector","type","Region","income"]].describe(include="all").T
"""))


# ── Section 3: FCY Cohort ──────────────────────────────────────────────────
cells.append(md("## 3 — FCY Cohort Assignment"))

cells.append(code("""\
df = fs.add_fcy_cohort(df)

cohort_stats = (df.groupby("fcy_cohort", observed=True)
                .agg(n=("target","count"), distress=("target","sum"))
                .assign(rate=lambda x: (x["distress"]/x["n"]*100).round(1))
                .reset_index())
print(cohort_stats.to_string(index=False))
"""))

cells.append(md("""\
**2015–2017 reporting-lag caveat:** Near-zero distress in the test cohort reflects that PPI data
for recently-closed projects takes several years to update in the database. This is *not* a genuine
signal that 2015–2017 projects are risk-free. If the model scores anomalously high AUC on the test
set, treat it as a reporting-lag artefact, not genuine lift — and document it as such.
"""))


# ── Section 4: WGI Enrichment ─────────────────────────────────────────────
cells.append(md("## 4 — WGI Governance Enrichment"))

cells.append(code("""\
wgi = fs.download_wgi()
print(f"WGI cache: {wgi.shape[0]:,} rows × {wgi.shape[1]} cols")
print("Indicators:", list(fs.WGI_SHEETS.values()))
wgi.head(3)
"""))

cells.append(code("""\
df = fs.join_wgi(df, wgi)
wgi_cols = list(fs.WGI_SHEETS.values())
fill_rates = df[wgi_cols].notna().mean() * 100
print("WGI fill rates by indicator (% of projects with data):")
print(fill_rates.round(1).to_string())
"""))

cells.append(md("""\
**WGI matching:** exact country name normalization → fuzzy match (`difflib.get_close_matches`) →
regional mean imputation for the ~13 unmatched projects. Coverage is 86%; the 14% gap is mainly
pre-1996 FCY projects (WGI series starts 1996 — those projects get the regional mean for their year).
"""))


# ── Section 5: Macro Enrichment ────────────────────────────────────────────
cells.append(md("## 5 — Macro Enrichment: Commodity Prices + Treasury Rate"))

cells.append(code("""\
ps = fs.download_pinksheet()
df = fs.join_pinksheet(df, ps)
df = fs.join_treasury(df)

print("\\nMacro feature fill rates:")
for col in ["energy_idx", "metals_idx", "treasury_10y"]:
    pct = df[col].notna().mean() * 100
    print(f"  {col}: {pct:.1f}%")
"""))

cells.append(md("""\
**Sources:**
- `energy_idx` / `metals_idx`: World Bank Pink Sheet Monthly Indices, annual averages.
- `treasury_10y`: FRED GS10 US 10-Year Treasury yield, hardcoded annual averages 1990–2017.

**Rationale:** Commodity prices at FCY capture the input cost environment that sponsors used to
set investment size and financing assumptions. High steel/copper costs → elevated construction risk.
The 10Y Treasury anchors the risk-free rate in project finance cost-of-capital calculations.
"""))


# ── Section 6: Feature Encoding ────────────────────────────────────────────
cells.append(md("## 6 — Feature Encoding and Final Matrix"))

cells.append(code("""\
# Run the full pipeline (reads from parquet cache if already built)
train, test = fs.build_features(force=True)

print(f"Train: {train.shape[0]:,} projects × {train.shape[1]} columns")
print(f"Test:  {test.shape[0]:,} projects × {test.shape[1]} columns")
"""))

cells.append(md("### 6a — Feature Groups"))
cells.append(code("""\
feature_cols = [c for c in train.columns if c not in ["target", "FCY"]]

wgi_names = list(fs.WGI_SHEETS.values())
groups = {
    "WGI governance":         [c for c in feature_cols if any(c.startswith(w) or c == f"missing_{w}" for w in wgi_names)],
    "Macro (commodity/rate)": [c for c in feature_cols if c in ["energy_idx","metals_idx","treasury_10y"]],
    "Categorical one-hot":    [c for c in feature_cols if any(c.startswith(p) for p in
                                ["sector_","ssector_","type_","Region_","income_","GGC_","CAM_",
                                 "bid_crit_","MLS_","BS_","fcy_cohort_"])],
    "Interaction terms":      [c for c in feature_cols if any(c.startswith(p) for p in
                                ["cam_x_sect_","sect_x_type_","reg_x_coh_"])],
}
# Numeric / binary = everything else
covered = set(c for v in groups.values() for c in v)
groups["Numeric / binary"] = [c for c in feature_cols if c not in covered]

print("Feature groups:")
for grp, cols in groups.items():
    print(f"  {grp:30s}: {len(cols):3d}")
print(f"  {'TOTAL':30s}: {sum(len(v) for v in groups.values())}")
"""))

cells.append(md("### 6b — Missingness Check"))
cells.append(code("""\
missing = train[feature_cols].isna().sum()
if missing.sum() == 0:
    print("No missing values in feature matrix (all imputed).")
else:
    print("Columns with missing values:")
    print(missing[missing > 0].sort_values(ascending=False).to_string())
"""))

cells.append(md("### 6c — Class Balance: Train vs Test"))
cells.append(code("""\
print(f"TRAIN (FCY ≤ 2014): n={len(train):,}  distress={int(train['target'].sum())}  ({train['target'].mean()*100:.1f}%)")
print(f"TEST  (FCY 2015-2017): n={len(test):,}   distress={int(test['target'].sum())}   ({test['target'].mean()*100:.1f}%)")
print()
train_rate = train["target"].mean()
print(f"Calibration baseline: constant prediction = {train_rate*100:.1f}% (train distress rate)")
print(f"  Train Brier score: {((train['target'] - train_rate)**2).mean():.4f}")
print(f"  Test  Brier score: {((test['target'] - train_rate)**2).mean():.4f}")
print()
print("AUC random baseline: 0.5000 (by definition)")
"""))

cells.append(md("""\
**Class balance summary:**

| Split | n | Distress | Rate |
|---|---|---|---|
| Train (FCY ≤ 2014) | 8,435 | 608 | 7.2% |
| Test (FCY 2015–2017) | 1,048 | 1 | 0.1% |

**Primary evaluation metric:** 5-fold stratified cross-validation on the training set (AUC + Brier score).
The test set provides an out-of-time ranking check, but its near-zero distress rate makes precision/recall
uninformative — document this explicitly in `03_modeling.ipynb`.
"""))


# ── Section 7: Numeric Feature Summary ─────────────────────────────────────
cells.append(md("## 7 — Numeric Feature Distributions (Train Set)"))
cells.append(code("""\
numeric_feats = [c for c in ["log_investment", "period", "private", "treasury_10y",
                 "energy_idx", "metals_idx",
                 "wgi_gov_effectiveness", "wgi_rule_of_law", "wgi_control_corruption"]
                 if c in train.columns]
train[numeric_feats].describe().round(3)
"""))


# ── Section 8: Correlation with Target ─────────────────────────────────────
cells.append(md("## 8 — Numeric Feature Correlation with Target"))
cells.append(code("""\
import matplotlib.pyplot as plt
from pathlib import Path

corr_with_target = (train[numeric_feats + ["target"]]
                    .corr()["target"]
                    .drop("target")
                    .sort_values())

fig, ax = plt.subplots(figsize=(7, 4))
colors = ["#d73027" if v < 0 else "#4575b4" for v in corr_with_target]
corr_with_target.plot(kind="barh", ax=ax, color=colors)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("Pearson r with target (distress=1)")
ax.set_title("Numeric feature correlation with distress outcome (train set)")
plt.tight_layout()

fig_path = ROOT / "notebooks" / "figures" / "02_numeric_correlations.png"
fig_path.parent.mkdir(exist_ok=True)
plt.savefig(fig_path, dpi=150)
plt.show()
print(f"Saved: {fig_path}")
"""))

cells.append(md("""\
**Key observations:**
- WGI governance scores (rule of law, control of corruption, government effectiveness) show **negative** correlation with distress — better governance → lower probability of project failure. Consistent with Hypothesis 3 (governance amplifier).
- `log_investment` shows a small positive correlation — larger projects carry more complexity risk.
- `treasury_10y` is positively correlated — higher rates at FCY increase debt service burden.
- Macro commodity indices show near-zero bivariate correlation (signal may be nonlinear or sector-conditional).
"""))


# ── Closing ────────────────────────────────────────────────────────────────
cells.append(md("""---
## Summary

Feature engineering complete. Output files:

| File | Rows | Columns | Distress rate |
|---|---|---|---|
| `data/processed/features_train.parquet` | 8,435 | 154 | 7.2% |
| `data/processed/features_test.parquet` | 1,048 | 154 | 0.1% (lag artefact) |

**Feature matrix (154 features):**
- 13 numeric/binary (log_investment, period, private, IDA eligibility, unsolicited flag, sector/greenfield/PPP binaries, WGI missingness indicators)
- 12 WGI governance scores (6 indicators × raw + missingness flag)
- 3 macro (energy index, metals index, 10Y Treasury at FCY)
- 57 categorical one-hot dummies (sector, subsector, type, region, income, CAM, bid_crit, GGC, MLS, BS, fcy_cohort)
- 69 interaction term dummies (CAM × sector, sector × type, region × fcy_cohort)

**Next:** `03_modeling.ipynb` — XGBoost/LightGBM classifier, 5-fold stratified CV, Platt scaling calibration.
"""))


# ── Write notebook ─────────────────────────────────────────────────────────
nb.cells = cells

with open(NB_PATH, "w") as f:
    nbf.write(nb, f)

print(f"Wrote {NB_PATH}")
