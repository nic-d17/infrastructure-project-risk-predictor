"""
Download and convert the World Bank Private Participation in Infrastructure (PPI) database.
Source: https://ppi.worldbank.org/en/ppidata
Full .dta file covers 1990-2024, ~6,400+ projects across 137 low/middle-income countries.
"""

import os
import requests
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PPI_DTA_URL = "https://www.worldbank.org/content/dam/PPI/documents/2024-PPI-Full-DTA.dta"
DTA_PATH = RAW_DIR / "ppi_2024_full.dta"
CSV_PATH = RAW_DIR / "ppi_2024_full.csv"


def download_ppi(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if DTA_PATH.exists() and not force:
        print(f"Already downloaded: {DTA_PATH}")
        return DTA_PATH

    print(f"Downloading PPI database from World Bank...")
    response = requests.get(PPI_DTA_URL, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    with open(DTA_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 64):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {pct:.1f}%  ({downloaded / 1e6:.1f} MB / {total / 1e6:.1f} MB)", end="")
    print(f"\nSaved to {DTA_PATH}")
    return DTA_PATH


def convert_to_csv(force: bool = False) -> pd.DataFrame:
    if CSV_PATH.exists() and not force:
        print(f"Already converted: {CSV_PATH}")
        return pd.read_csv(CSV_PATH, low_memory=False)

    print("Reading .dta file with pandas...")
    df = pd.read_stata(DTA_PATH, convert_categoricals=False)
    print(f"  Shape: {df.shape}")
    print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved CSV: {CSV_PATH}")
    return df


def inspect(df: pd.DataFrame) -> None:
    print("\n--- PPI column inventory ---")
    for col in df.columns:
        n_missing = df[col].isna().sum()
        print(f"  {col:40s}  dtype={str(df[col].dtype):10s}  missing={n_missing}/{len(df)}")

    cost_cols = [c for c in df.columns if any(k in c.lower() for k in
                 ["invest", "cost", "commit", "actual", "capex", "usd", "million"])]
    if cost_cols:
        print(f"\nCost-related columns: {cost_cols}")
        print(df[cost_cols].describe())


if __name__ == "__main__":
    download_ppi()
    df = convert_to_csv()
    inspect(df)
