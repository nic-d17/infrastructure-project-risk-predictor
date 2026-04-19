"""
Decode PPI STATA value labels and produce a fully human-readable CSV.

The raw PPI .dta file stores all categorical fields as integers.
pandas read_stata with convert_categoricals=True recovers the label strings.
This script also prints a complete data dictionary and saves a decoded CSV.

Output:
  data/raw/ppi_2024_decoded.csv         — all categoricals as strings
  data/interim/ppi_data_dictionary.txt  — field-by-field value label map
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
INTERIM_DIR = Path(__file__).resolve().parents[2] / "data" / "interim"
DTA_PATH = RAW_DIR / "ppi_2024_full.dta"
DECODED_CSV = RAW_DIR / "ppi_2024_decoded.csv"
DICT_PATH = INTERIM_DIR / "ppi_data_dictionary.txt"

# Columns that remain numeric (not value-labeled categoricals)
NUMERIC_COLS = {
    "ID", "IY", "FCY", "FCM", "period", "VDGS", "TIGS", "VIGS",
    "private", "fees", "physical", "investment", "capacity", "pcapacity",
    "numberb", "PRS", "FundingYear", "debt", "equity",
    "c_debt", "m_debt", "b_debt", "i_debt", "p_debt", "intl_debt", "l_debt",
    "CPI2024", "CapacityDollar", "shareborder",
}


def decode(force: bool = False) -> pd.DataFrame:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    if DECODED_CSV.exists() and not force:
        print(f"Already decoded: {DECODED_CSV}")
        return pd.read_csv(DECODED_CSV, low_memory=False)

    print("Reading .dta with value labels...")
    df = pd.read_stata(DTA_PATH, convert_categoricals=True)
    print(f"  Shape: {df.shape}")

    df.to_csv(DECODED_CSV, index=False)
    print(f"Saved decoded CSV: {DECODED_CSV}")
    return df


def write_data_dictionary(df: pd.DataFrame) -> None:
    lines = ["PPI 2024 — Data Dictionary", "=" * 60, ""]

    # Field descriptions based on PPI methodology documentation
    field_meta = {
        "ID":           "Unique project identifier",
        "IY":           "Income year (World Bank income classification year)",
        "country":      "Country name",
        "Region":       "World Bank region (East Asia, South Asia, LAC, SSA, MENA, ECA)",
        "income":       "World Bank income group (Low, Lower-middle, Upper-middle)",
        "IDA":          "IDA-eligible country (0/1)",
        "FCY":          "Financial close year",
        "FCM":          "Financial close month",
        "type":         "Project type (Greenfield / Brownfield / Divestiture / Management and lease contract)",
        "stype":        "Project sub-type",
        "status_n":     "Project status: Active / Distressed / Cancelled / Concluded — PRIMARY TARGET",
        "sector":       "Sector: Energy / Transport / Water and sewerage / ICT / Municipal Solid Waste",
        "ssector":      "Sub-sector (e.g., Toll road, Power generation, Port)",
        "Segment":      "Segment within sub-sector",
        "period":       "Project construction/concession period (years)",
        "GGC":          "Government guarantee on commercial risk (0/1)",
        "VDGS":         "Value of direct government support (USD m)",
        "TIGS":         "Total indirect government support",
        "VIGS":         "Value of indirect government support",
        "private":      "Private investment share (%)",
        "fees":         "User fees / tariff revenue (USD m/yr)",
        "physical":     "Physical capacity (units depend on sector: MW, km, m3/day, etc.)",
        "investment":   "Total investment commitments at financial close (USD m) — primary cost field",
        "capacity":     "Capacity unit type code",
        "pcapacity":    "Per-unit capacity cost (USD / capacity unit)",
        "technol":      "Technology type (generation technology for energy projects)",
        "bid_crit":     "Primary bidding criterion (tariff, subsidy, upfront payment, etc.)",
        "CAM":          "Contract award method (competitive / direct / other)",
        "numberb":      "Number of bidders at award",
        "PRS":          "PPI Research Score / data quality score",
        "OSR":          "Off-system record flag",
        "Description":  "Free-text project description",
        "FundingYear":  "Year additional public funding was provided",
        "debt":         "Total debt finance (USD m)",
        "equity":       "Total equity finance (USD m or breakdown)",
        "c_debt":       "Commercial bank debt (USD m)",
        "m_debt":       "Multilateral bank debt (USD m)",
        "b_debt":       "Bilateral agency debt (USD m)",
        "i_debt":       "IFC / IFI debt (USD m)",
        "p_debt":       "Pension / institutional debt (USD m)",
        "intl_debt":    "International bond debt (USD m)",
        "l_debt":       "Local currency debt (USD m)",
        "UP":           "Unsolicited proposal (0/1)",
        "PublicDisclosure": "Public disclosure level",
        "bordercountries":  "Cross-border project flag",
        "shareborder":  "Border country count",
        "CPI2024":      "CPI index value (2024 base) for investment deflation",
        "Check":        "Data quality check flag (yay/nay/blank)",
        "CapacityDollar": "Investment per capacity unit (USD)",
        "name":         "Project name",
        "MLS":          "Multilateral support flag",
        "BS":           "Bond-supported flag",
        "PPP":          "Public-private partnership flag (0/1)",
    }

    for col in df.columns:
        desc = field_meta.get(col, "See PPI methodology documentation")
        dtype = str(df[col].dtype)
        n_missing = df[col].isna().sum()
        lines.append(f"{col}")
        lines.append(f"  Description : {desc}")
        lines.append(f"  dtype       : {dtype}")
        lines.append(f"  missing     : {n_missing}/{len(df)} ({n_missing/len(df)*100:.1f}%)")
        if col not in NUMERIC_COLS and df[col].dtype == "object":
            vals = df[col].value_counts().head(10)
            lines.append(f"  top values  : {dict(vals)}")
        elif col not in NUMERIC_COLS and df[col].dtype.name == "category":
            vals = df[col].value_counts()
            lines.append(f"  all values  : {dict(vals)}")
        lines.append("")

    DICT_PATH.write_text("\n".join(lines))
    print(f"Data dictionary written: {DICT_PATH}")


def print_summary(df: pd.DataFrame) -> None:
    print("\n=== PPI DECODED FIELD SUMMARY ===")

    print("\n--- status_n (TARGET VARIABLE) ---")
    print(df["status_n"].value_counts())
    print(f"  Distress rate: {(df['status_n'].isin(['Distressed','Cancelled'])).mean()*100:.1f}%")

    print("\n--- sector ---")
    print(df["sector"].value_counts())

    print("\n--- type (project type) ---")
    print(df["type"].value_counts())

    print("\n--- Region ---")
    print(df["Region"].value_counts())

    print("\n--- income group ---")
    print(df["income"].value_counts())

    print("\n--- FCY (financial close year) distribution ---")
    fcy = df["FCY"].value_counts().sort_index()
    print(f"  Range: {fcy.index.min()} – {fcy.index.max()}")
    print(f"  Pre-2015: {(df['FCY'] < 2015).sum()} projects  |  2015+: {(df['FCY'] >= 2015).sum()} projects")

    print("\n--- investment (USD m) ---")
    print(df["investment"].describe())

    print("\n--- GGC (govt guarantee) ---")
    print(df["GGC"].value_counts())

    print("\n--- CAM (contract award method) ---")
    print(df["CAM"].value_counts())

    print("\n--- PPP flag ---")
    print(df["PPP"].value_counts())


if __name__ == "__main__":
    df = decode()
    write_data_dictionary(df)
    print_summary(df)
