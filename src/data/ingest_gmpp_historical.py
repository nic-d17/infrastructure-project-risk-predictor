"""
Download UK GMPP data for all available years (2015-2023) and consolidate.

URL pattern (confirmed for 2024): /government/publications/[dept]-government-major-projects-portfolio-data-[year]
The same 21 departments are tried for each year; missing department-years are silently skipped.

Output: data/raw/gmpp_[year]_all_departments.csv for each year
        data/raw/gmpp_all_years.csv — full longitudinal stack
"""

import re
import time
import requests
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
BASE_URL = "https://www.gov.uk"
HEADERS = {"User-Agent": "Mozilla/5.0 (research data download)"}

YEARS = list(range(2015, 2024))  # 2015–2023; 2024 already downloaded

DEPT_SLUGS = [
    "co", "dbt", "dcms", "defra", "desnz", "dfe", "dft", "dhsc",
    "dluhc", "dsit", "dwp", "fcdo", "hmlr", "hmrc", "hmt", "ho",
    "mod", "moj", "nca", "ons", "voa",
    # older department names that predate 2024 restructure
    "beis", "decc", "bis", "dh", "dclg", "dfid",
]


def _find_csv_url(dept: str, year: int) -> str | None:
    page_path = f"/government/publications/{dept}-government-major-projects-portfolio-data-{year}"
    url = BASE_URL + page_path
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        matches = re.findall(
            r'https://assets\.publishing\.service\.gov\.uk[^"\'<>\s]+\.csv',
            r.text
        )
        if matches:
            return matches[0]
    except Exception:
        pass
    return None


def download_year(year: int, force: bool = False) -> pd.DataFrame | None:
    out_path = RAW_DIR / f"gmpp_{year}_all_departments.csv"
    if out_path.exists() and not force:
        print(f"  {year}: already on disk ({out_path.name})")
        return pd.read_csv(out_path, low_memory=False)

    frames = []
    for dept in DEPT_SLUGS:
        csv_url = _find_csv_url(dept, year)
        if not csv_url:
            continue
        try:
            r = requests.get(csv_url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            from io import StringIO
            df_dept = pd.read_csv(StringIO(r.text))
            df_dept["_source_dept"] = dept.upper()
            df_dept["_source_year"] = year
            frames.append(df_dept)
            print(f"    [{year}][{dept.upper():6s}] {len(df_dept)} rows")
        except Exception as e:
            print(f"    [{year}][{dept.upper():6s}] FAILED: {e}")
        time.sleep(0.3)

    if not frames:
        print(f"  {year}: no data found")
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_path, index=False)
    print(f"  {year}: {len(frames)} depts, {len(combined)} rows → {out_path.name}")
    return combined


def build_longitudinal(force: bool = False) -> pd.DataFrame:
    out_path = RAW_DIR / "gmpp_all_years.csv"
    if out_path.exists() and not force:
        print(f"Longitudinal file already exists: {out_path}")
        return pd.read_csv(out_path, low_memory=False)

    all_frames = []

    # Include 2024 already downloaded
    path_2024 = RAW_DIR / "gmpp_2024_all_departments.csv"
    if path_2024.exists():
        df_2024 = pd.read_csv(path_2024, low_memory=False)
        all_frames.append(df_2024)

    for year in YEARS:
        df_year = download_year(year)
        if df_year is not None:
            all_frames.append(df_year)

    if not all_frames:
        raise RuntimeError("No GMPP data available.")

    longitudinal = pd.concat(all_frames, ignore_index=True)
    longitudinal.to_csv(out_path, index=False)
    print(f"\nLongitudinal stack: {len(longitudinal)} rows across {longitudinal['_source_year'].nunique()} years")
    return longitudinal


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = build_longitudinal()
    print("\n=== LONGITUDINAL SUMMARY ===")
    print(df.groupby("_source_year")["_source_dept"].count().rename("project_rows"))
