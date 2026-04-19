"""
Download and consolidate UK Government Major Projects Portfolio (GMPP) data.

Each department publishes a separate CSV annually via gov.uk.
This script fetches all 21 departmental pages for the 2024 release,
extracts the CSV download link from each, downloads, and concatenates.

Source collection: https://www.gov.uk/government/collections/major-projects-data
Schema (per 2024 release):
  GMPP ID, Project Name, Department, Annual Report Category,
  Description/Aims, IPA Delivery Confidence Assessment,
  Project Start/End Date, Financial Year Baseline (£m),
  Financial Year Forecast (£m), Financial Year Variance (%),
  TOTAL Baseline Whole Life Costs (£m)
"""

import re
import time
import requests
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUT_CSV = RAW_DIR / "gmpp_2024_all_departments.csv"

BASE_URL = "https://www.gov.uk"

# All 21 departmental publication pages for 2024 (published 2025-01-16)
DEPT_PAGES_2024 = {
    "CO":    "/government/publications/co-government-major-projects-portfolio-data-2024",
    "DBT":   "/government/publications/dbt-government-major-projects-portfolio-data-2024",
    "DCMS":  "/government/publications/dcms-government-major-projects-portfolio-data-2024",
    "DEFRA": "/government/publications/defra-government-major-projects-portfolio-data-2024",
    "DESNZ": "/government/publications/desnz-government-major-projects-portfolio-data-2024",
    "DFE":   "/government/publications/dfe-government-major-projects-portfolio-data-2024",
    "DFT":   "/government/publications/dft-government-major-projects-portfolio-data-2024",
    "DHSC":  "/government/publications/dhsc-government-major-projects-portfolio-data-2024",
    "DLUHC": "/government/publications/dluhc-government-major-projects-portfolio-data-2024",
    "DSIT":  "/government/publications/dsit-government-major-projects-portfolio-data-2024",
    "DWP":   "/government/publications/dwp-government-major-projects-portfolio-data-2024",
    "FCDO":  "/government/publications/fcdo-government-major-projects-portfolio-data-2024",
    "HMLR":  "/government/publications/hmlr-government-major-projects-portfolio-data-2024",
    "HMRC":  "/government/publications/hmrc-government-major-projects-portfolio-data-2024",
    "HMT":   "/government/publications/hmt-government-major-projects-portfolio-data-2024",
    "HO":    "/government/publications/ho-government-major-projects-portfolio-data-2024",
    "MOD":   "/government/publications/mod-government-major-projects-portfolio-data-2024",
    "MOJ":   "/government/publications/moj-government-major-projects-portfolio-data-2024",
    "NCA":   "/government/publications/nca-government-major-projects-portfolio-data-2024",
    "ONS":   "/government/publications/ons-government-major-projects-portfolio-data-2024",
    "VOA":   "/government/publications/voa-government-major-projects-portfolio-data-2024",
}

# Known working direct CSV URLs (fallback if scraping fails)
KNOWN_CSV_URLS = {
    "HMT": "https://assets.publishing.service.gov.uk/media/6787e36ebca9366c9f56df7b/HMT_Government_Major_Projects_Portofolio_AR_Data_March_2024.csv",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (research data download)"}


def _find_csv_url(dept: str, page_path: str) -> str | None:
    if dept in KNOWN_CSV_URLS:
        return KNOWN_CSV_URLS[dept]

    url = BASE_URL + page_path
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        # Look for assets.publishing.service.gov.uk CSV links
        matches = re.findall(
            r'https://assets\.publishing\.service\.gov\.uk[^"\']+\.csv',
            r.text
        )
        if matches:
            return matches[0]
        # Fallback: look for any CSV attachment link
        matches = re.findall(r'href="([^"]+\.csv)"', r.text)
        if matches:
            return BASE_URL + matches[0] if matches[0].startswith("/") else matches[0]
    except Exception as e:
        print(f"  WARNING: failed to fetch {url}: {e}")
    return None


def download_all(force: bool = False) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_CSV.exists() and not force:
        print(f"Already consolidated: {OUTPUT_CSV}")
        return pd.read_csv(OUTPUT_CSV, low_memory=False)

    frames = []
    for dept, page_path in DEPT_PAGES_2024.items():
        print(f"  [{dept}] finding CSV link...", end=" ")
        csv_url = _find_csv_url(dept, page_path)
        if not csv_url:
            print("SKIPPED (no CSV found)")
            continue

        try:
            r = requests.get(csv_url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            from io import StringIO
            df_dept = pd.read_csv(StringIO(r.text))
            df_dept["_source_dept"] = dept
            df_dept["_source_year"] = 2024
            frames.append(df_dept)
            print(f"OK ({len(df_dept)} rows)")
        except Exception as e:
            print(f"FAILED ({e})")

        time.sleep(0.5)  # be polite to gov.uk

    if not frames:
        raise RuntimeError("No GMPP departmental files downloaded successfully.")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\nConsolidated {len(frames)} departments → {len(combined)} rows → {OUTPUT_CSV}")
    return combined


def inspect(df: pd.DataFrame) -> None:
    print("\n--- GMPP consolidated summary ---")
    print(f"  Total projects: {len(df)}")
    print(f"  Departments: {df['_source_dept'].nunique()}")
    print(f"  Columns: {list(df.columns)}")

    cost_cols = [c for c in df.columns if any(k in c.lower() for k in
                 ["baseline", "forecast", "variance", "cost", "benefit"])]
    print(f"\n  Cost/variance columns: {cost_cols}")

    if "Financial Year Variance (%)" in df.columns:
        var = pd.to_numeric(df["Financial Year Variance (%)"], errors="coerce")
        print(f"\n  Financial Year Variance (%) stats:")
        print(var.describe())


if __name__ == "__main__":
    df = download_all()
    inspect(df)
