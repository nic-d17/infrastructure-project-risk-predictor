# Preprocessing script for World Bank PPI data.
#
# KEY DESIGN DECISION — 7-year exposure window:
#   Target = 1 if status_n in {Distressed, Cancelled}, else 0.
#   Projects with FCY > 2017 are DROPPED before any train/test split.
#
#   Rationale:
#   (1) Survival-time confounding: a project with FCY = 2021 that is still
#       "Active" in 2024 has only had 3 years to fail. Labeling it as a
#       negative (non-distress) case would be misleading — it simply hasn't
#       had enough exposure time. Including it would suppress the model's
#       estimated distress rate for recent cohorts.
#   (2) Infra PE alignment: typical hold periods are 5–10 years. Predicting
#       P(distress within 7 years) is directly actionable for a credit
#       committee underwriting a 7-year senior secured infrastructure loan.
#   (3) 2017 cutoff: projects closed in 2017 have had 7 full years of
#       exposure by the 2024 snapshot date. FCY = 2018 → only 6 years.
#
#   This decision is documented in docs/PROJECT_PLAN.md and CLAUDE.md.

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

EXPOSURE_CUTOFF_YEAR = 2017       # FCY > this value → dropped
SNAPSHOT_YEAR = 2024              # year the PPI snapshot was taken
MIN_EXPOSURE_YEARS = 7            # SNAPSHOT_YEAR - EXPOSURE_CUTOFF_YEAR

DISTRESS_STATUSES = {"Distressed", "Cancelled"}

# Temporal split boundary: train on FCY < this, test on this and above
TRAIN_TEST_SPLIT_YEAR = 2010


def load_decoded() -> pd.DataFrame:
    path = RAW_DIR / "ppi_2024_decoded.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run decode_ppi.py first: {path}")
    return pd.read_csv(path, low_memory=False)


def apply_exposure_filter(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df[df["FCY"] <= EXPOSURE_CUTOFF_YEAR].copy()
    n_dropped = n_before - len(df)
    print(f"Exposure filter (FCY <= {EXPOSURE_CUTOFF_YEAR}): "
          f"kept {len(df):,} / dropped {n_dropped:,} projects")
    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    df["target"] = df["status_n"].isin(DISTRESS_STATUSES).astype(int)
    rate = df["target"].mean()
    n_pos = df["target"].sum()
    print(f"Target: {n_pos:,} distressed/cancelled of {len(df):,} ({rate*100:.1f}%)")
    return df


def add_temporal_split(df: pd.DataFrame) -> pd.DataFrame:
    df["split"] = np.where(df["FCY"] < TRAIN_TEST_SPLIT_YEAR, "train", "test")
    counts = df.groupby("split")["target"].agg(["count", "sum", "mean"])
    counts.columns = ["n_projects", "n_distress", "distress_rate"]
    print("\nTemporal split:")
    print(counts.to_string())
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    # Only fields available at financial close with >50% coverage
    keep = [
        "ID", "FCY", "target", "split",
        # Categorical features (decoded strings)
        "sector", "ssector", "type", "Region", "income",
        "GGC", "CAM", "technol", "bid_crit", "status_n",
        "IDA", "UP", "PPP", "bordercountries",
        # Numeric features
        "investment",   # USD m — primary cost proxy
        "period",       # concession/construction period (years)
        "private",      # private investment share (%)
        "numberb",      # number of bidders
        "pcapacity",    # investment per capacity unit
        "PRS",          # data quality score
        "MLS", "BS",    # multilateral support, bond support flags
    ]
    available = [c for c in keep if c in df.columns]
    return df[available].copy()


def save(df: pd.DataFrame) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "ppi_model_ready.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved model-ready file: {out}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")
    return out


def run(force: bool = False) -> pd.DataFrame:
    out_path = PROCESSED_DIR / "ppi_model_ready.csv"
    if out_path.exists() and not force:
        print(f"Already preprocessed: {out_path}")
        return pd.read_csv(out_path)

    df = load_decoded()
    df = apply_exposure_filter(df)
    df = build_target(df)
    df = add_temporal_split(df)
    df = select_features(df)
    save(df)
    return df


if __name__ == "__main__":
    df = run(force=True)
    print("\n--- Model-ready dataset head ---")
    print(df.head(3).to_string())
