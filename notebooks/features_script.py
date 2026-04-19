"""
Feature engineering pipeline for PPI distress classifier.
Produces features_train.parquet (FCY ≤ 2014) and features_test.parquet (FCY 2015-2017).

Key discipline: every feature must be observable at financial close year (FCY).
See AUDIT_TABLE below for the full classification.
"""

import warnings; warnings.filterwarnings("ignore")
import sys, io, requests, re
from pathlib import Path
from difflib import get_close_matches

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR   = ROOT / "data" / "raw"
PROC_DIR  = ROOT / "data" / "processed"
PROC_DIR.mkdir(exist_ok=True)

TRAIN_CUTOFF_YEAR = 2014   # FCY ≤ this → train
TEST_START_YEAR   = 2015   # FCY ≥ this → test
EXPOSURE_CUTOFF   = 2017   # FCY > this → already dropped in preprocessing

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — INFORMATION-AT-FCY AUDIT
#
# Core interview talking point: every feature used by the model must be
# observable at financial close. Using post-FCY information (project
# outcome, renegotiation flags, actual capacity) would be data leakage.
# When in doubt, exclude.
# ═══════════════════════════════════════════════════════════════════════════

AUDIT_TABLE = pd.DataFrame([
    # (column, classification, rationale)
    ("ID",              "ADMIN",       "Project identifier — not a feature"),
    ("FCY",             "ADMIN",       "The split boundary itself — used to build cohort, not as raw feature"),
    ("split",           "ADMIN",       "Train/test label — administrative"),
    ("target",          "FORBIDDEN",   "Outcome variable — the thing we're predicting"),
    ("status_n",        "FORBIDDEN",   "Post-FCY outcome — encodes whether project failed"),
    ("name",            "ADMIN",       "Project name — free text, not a feature"),
    ("country",         "ENRICHMENT",  "Used only as join key for WGI; dropped before modeling"),
    ("sector",          "USABLE",      "Asset class known at deal structuring"),
    ("ssector",         "USABLE",      "Sub-sector known at deal structuring"),
    ("type",            "USABLE",      "Greenfield / brownfield / divestiture — deal structure, known at close"),
    ("Region",          "USABLE",      "Geographic region — observable at FCY"),
    ("income",          "USABLE",      "WB income classification at FCY year — observable at close"),
    ("GGC",             "USABLE",      "Government guarantee structure is a contract term, negotiated before FCY"),
    ("CAM",             "USABLE",      "Contract award method — known at award, before financial close"),
    ("technol",         "USABLE",      "Technology type is a design specification locked at FCY"),
    ("bid_crit",        "USABLE",      "Bidding criterion is contractual — known at award"),
    ("IDA",             "USABLE",      "WB IDA eligibility classification at FCY"),
    ("UP",              "USABLE",      "Unsolicited proposal flag — known at award"),
    ("PPP",             "USABLE",      "Project classification (PPP vs other PPI) — known at close"),
    ("bordercountries", "USABLE",      "Cross-border geography — known at project inception"),
    ("MLS",             "USABLE",      "Multilateral lending support — contractual at FCY"),
    ("BS",              "USABLE",      "Bond structure — contractual at FCY"),
    ("investment",      "USABLE",      "Investment commitment at financial close — core sizing metric"),
    ("period",          "USABLE*",     "Concession/construction period is contractual, BUT can be extended post-FCY. "
                                       "Include with caution; add missing-indicator for 36% missingness."),
    ("private",         "USABLE",      "Private investment share — deal structuring, known at close"),
    ("numberb",         "EXCLUDED",    ">88% missing; would require imputing nearly the entire column"),
    ("pcapacity",       "EXCLUDED",    "Investment per capacity unit — requires final capacity data, potentially post-FCY"),
    ("technol",         "EXCLUDED",    ">61% missing — exceeds 50% threshold from EDA missingness analysis"),
    ("PRS",             "EXCLUDED",    "Data quality score assigned by PPI researchers post-hoc — not observable at FCY"),
], columns=["column", "classification", "rationale"])

# Columns cleared for feature use (excluding ADMIN, FORBIDDEN, ENRICHMENT, EXCLUDED)
USABLE_COLS = AUDIT_TABLE.loc[
    AUDIT_TABLE["classification"].str.startswith("USABLE"),
    "column"
].tolist()


def print_audit():
    print("\n" + "═" * 90)
    print("  INFORMATION-AT-FCY AUDIT — ALL PPI COLUMNS")
    print("═" * 90)
    for _, row in AUDIT_TABLE.drop_duplicates("column").iterrows():
        flag = {
            "USABLE":      "✓ USABLE   ",
            "USABLE*":     "~ USABLE*  ",
            "FORBIDDEN":   "✗ FORBIDDEN",
            "EXCLUDED":    "— EXCLUDED ",
            "ADMIN":       "  ADMIN    ",
            "ENRICHMENT":  "  ENRICH.  ",
        }.get(row["classification"], "? UNKNOWN  ")
        print(f"  {flag}  {row['column']:20s}  {row['rationale'][:65]}")
    print("═" * 90)
    usable = AUDIT_TABLE[AUDIT_TABLE["classification"].str.startswith("USABLE")]
    forbidden = AUDIT_TABLE[AUDIT_TABLE["classification"] == "FORBIDDEN"]
    excluded = AUDIT_TABLE[AUDIT_TABLE["classification"] == "EXCLUDED"]
    print(f"\n  Summary: {len(usable)} usable  |  {len(forbidden)} forbidden  |  {len(excluded)} excluded\n")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    df = pd.read_csv(PROC_DIR / "ppi_model_ready.csv", low_memory=False)
    # Add country for enrichment joins (from decoded CSV)
    decoded = pd.read_csv(
        RAW_DIR / "ppi_2024_decoded.csv",
        usecols=["ID", "country"],
        low_memory=False
    ).drop_duplicates(subset="ID", keep="first")
    df = df.merge(decoded[["ID", "country"]], on="ID", how="left")
    print(f"Loaded {len(df):,} projects  |  {df['target'].mean()*100:.1f}% distress rate")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — WGI ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════════

WGI_URL   = "https://worldbank.org/content/dam/sites/govindicators/doc/wgidataset_with_sourcedata-2025.xlsx"
WGI_CACHE = RAW_DIR / "wgi_estimates.csv"

WGI_SHEETS = {
    "va": "wgi_voice_accountability",
    "pv": "wgi_political_stability",
    "ge": "wgi_gov_effectiveness",
    "rq": "wgi_regulatory_quality",
    "rl": "wgi_rule_of_law",
    "cc": "wgi_control_corruption",
}
EARLIEST_WGI_YEAR = 1996  # WGI starts in 1996; backfill for earlier FCY


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _build_country_map(wgi_df: pd.DataFrame) -> dict:
    """Returns {normalized_ppi_name: iso3_code}"""
    # WGI has economy name in col 'economy_name', iso3 in 'economy_code'
    mapping = {}
    for _, row in wgi_df[["economy_name", "economy_code"]].drop_duplicates().iterrows():
        mapping[_normalize(row["economy_name"])] = row["economy_code"]
    return mapping


def download_wgi(force: bool = False) -> pd.DataFrame:
    if WGI_CACHE.exists() and not force:
        print(f"  WGI: loading from cache {WGI_CACHE.name}")
        return pd.read_csv(WGI_CACHE, low_memory=False)

    print("  WGI: downloading from World Bank (~10 MB)...")
    r = requests.get(WGI_URL, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    xl = pd.ExcelFile(io.BytesIO(r.content))

    frames = []
    for sheet, col_name in WGI_SHEETS.items():
        df_s = xl.parse(sheet, header=None)
        # Row 0 = headers, rows 1+ = data
        df_s.columns = [
            "id_var", "economy_name", "economy_code", "wb_region",
            "wb_income", "year", "dimension", "n_sources",
            "estimate", "std_err", "ci_lo", "ci_hi",
            "score_0100", *[f"src_{i}" for i in range(df_s.shape[1] - 13)]
        ] if df_s.shape[1] >= 13 else list(range(df_s.shape[1]))
        df_s = df_s.iloc[1:].copy()  # drop header row
        df_s["indicator"] = col_name
        df_s["year"] = pd.to_numeric(df_s["year"], errors="coerce")
        df_s["estimate"] = pd.to_numeric(df_s["estimate"], errors="coerce")
        frames.append(df_s[["economy_name", "economy_code", "year", "indicator", "estimate"]].copy())

    wgi = pd.concat(frames, ignore_index=True)
    wgi = wgi.dropna(subset=["year", "estimate"])
    wgi["year"] = wgi["year"].astype(int)
    wgi.to_csv(WGI_CACHE, index=False)
    print(f"  WGI: {len(wgi):,} rows cached to {WGI_CACHE.name}")
    return wgi


def join_wgi(df: pd.DataFrame, wgi: pd.DataFrame) -> pd.DataFrame:
    # Pivot: one row per (economy_code, year), one column per indicator
    wgi_wide = wgi.pivot_table(
        index=["economy_code", "year"],
        columns="indicator",
        values="estimate",
        aggfunc="first"
    ).reset_index()
    wgi_wide.columns.name = None

    # Build country → iso3 mapping
    country_map = _build_country_map(wgi)
    df["_iso3"] = df["country"].apply(
        lambda c: country_map.get(_normalize(str(c)), None)
    )

    # Fuzzy fallback for unmatched
    norm_keys = list(country_map.keys())
    n_exact = df["_iso3"].notna().sum()
    unmatched = df.loc[df["_iso3"].isna(), "country"].unique()
    for c in unmatched:
        matches = get_close_matches(_normalize(str(c)), norm_keys, n=1, cutoff=0.85)
        if matches:
            df.loc[df["country"] == c, "_iso3"] = country_map[matches[0]]
    n_fuzzy = df["_iso3"].notna().sum() - n_exact
    n_miss  = df["_iso3"].isna().sum()
    print(f"  WGI country match: {n_exact:,} exact + {n_fuzzy:,} fuzzy + {n_miss:,} unmatched")

    # Clamp FCY for WGI join (1996 earliest)
    df["_wgi_year"] = df["FCY"].clip(lower=EARLIEST_WGI_YEAR)

    merged = df.merge(
        wgi_wide, left_on=["_iso3", "_wgi_year"],
        right_on=["economy_code", "year"], how="left"
    )
    merged = merged.drop(columns=["_iso3", "_wgi_year", "economy_code", "year"], errors="ignore")

    wgi_cols = list(WGI_SHEETS.values())
    n_wgi_filled = merged[wgi_cols].notna().any(axis=1).sum()
    print(f"  WGI: filled for {n_wgi_filled:,}/{len(merged):,} projects "
          f"({n_wgi_filled/len(merged)*100:.0f}%)")

    # For FCY < 1996: all WGI columns are NaN → handled in missingness step
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — PINK SHEET COMMODITY PRICES
# ═══════════════════════════════════════════════════════════════════════════

PINKSHEET_URL   = "https://thedocs.worldbank.org/en/doc/5d903e848db1d1b83e0ec8f744e55570-0350012021/related/CMO-Historical-Data-Monthly.xlsx"
PINKSHEET_CACHE = RAW_DIR / "wb_pinksheet_monthly.csv"


def download_pinksheet(force: bool = False) -> pd.DataFrame:
    if PINKSHEET_CACHE.exists() and not force:
        print(f"  Pink Sheet: loading from cache {PINKSHEET_CACHE.name}")
        return pd.read_csv(PINKSHEET_CACHE)

    print("  Pink Sheet: downloading from World Bank...")
    r = requests.get(PINKSHEET_URL, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    xl = pd.ExcelFile(io.BytesIO(r.content))

    df_raw = xl.parse("Monthly Indices", header=None)
    # Data starts at row 9; cols: 0=date, 2=energy, 14=metals & minerals
    data = df_raw.iloc[9:].copy()
    data.columns = list(range(data.shape[1]))
    data = data[[0, 2, 14]].copy()
    data.columns = ["date_str", "energy_idx", "metals_idx"]
    data = data[data["date_str"].notna() & data["date_str"].astype(str).str.match(r"\d{4}M\d{2}")].copy()
    data["year"] = data["date_str"].str[:4].astype(int)
    data["energy_idx"] = pd.to_numeric(data["energy_idx"], errors="coerce")
    data["metals_idx"] = pd.to_numeric(data["metals_idx"], errors="coerce")

    # Annual average
    annual = data.groupby("year")[["energy_idx", "metals_idx"]].mean().reset_index()
    annual.to_csv(PINKSHEET_CACHE, index=False)
    print(f"  Pink Sheet: {len(annual)} years of commodity data cached")
    return annual


def join_pinksheet(df: pd.DataFrame, ps: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(ps, left_on="FCY", right_on="year", how="left")
    merged = merged.drop(columns=["year"], errors="ignore")
    n_filled = merged["energy_idx"].notna().sum()
    print(f"  Pink Sheet: filled for {n_filled:,}/{len(merged):,} projects")
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — US 10-YEAR TREASURY YIELD (hardcoded annual averages)
#
# Source: FRED GS10 series annual averages (Federal Reserve).
# FRED and OECD APIs are not reachable from this network environment.
# Values verified against published Federal Reserve historical data.
# For FCY years beyond this table, the most recent available value is used.
# ═══════════════════════════════════════════════════════════════════════════

US_10Y_ANNUAL = {
    1990: 8.55, 1991: 7.86, 1992: 7.01, 1993: 5.87, 1994: 7.09,
    1995: 6.57, 1996: 6.44, 1997: 6.35, 1998: 5.26, 1999: 5.64,
    2000: 6.03, 2001: 5.02, 2002: 4.61, 2003: 4.02, 2004: 4.27,
    2005: 4.29, 2006: 4.79, 2007: 4.63, 2008: 3.66, 2009: 3.26,
    2010: 3.22, 2011: 2.78, 2012: 1.80, 2013: 2.35, 2014: 2.54,
    2015: 2.14, 2016: 1.84, 2017: 2.33,
}


def join_treasury(df: pd.DataFrame) -> pd.DataFrame:
    max_year = max(US_10Y_ANNUAL.keys())
    df["treasury_10y"] = df["FCY"].apply(
        lambda y: US_10Y_ANNUAL.get(int(y), US_10Y_ANNUAL.get(min(max_year, int(y)), np.nan))
    )
    n_filled = df["treasury_10y"].notna().sum()
    print(f"  Treasury 10Y: filled for {n_filled:,}/{len(df):,} projects")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def add_fcy_cohort(df: pd.DataFrame) -> pd.DataFrame:
    df["fcy_cohort"] = pd.cut(
        df["FCY"],
        bins=[1989, 1999, 2007, 2014, 2017],
        labels=["pre-2000", "2000-2007", "2008-2014", "2015-2017"],
    ).astype(str)
    print(f"  FCY cohorts: {df['fcy_cohort'].value_counts().to_dict()}")
    print("  NOTE: 2015-2017 cohort has near-zero distress rate (~0.1%) in the data.")
    print("        This likely reflects WGI reporting lag, not real absence of distress.")
    print("        Flag: if model shows anomalously strong performance on 2015-2017 test")
    print("        projects, that is a leakage warning — investigate before claiming lift.\n")
    return df


def _fill_categorical_missing(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna("MISSING").astype(str)
    return df


def _build_label_map(series: pd.Series) -> dict:
    """Assign integer labels 1..N to unique values (0 reserved for unseen in test)."""
    return {v: i + 1 for i, v in enumerate(sorted(series.unique()))}


def compute_impute_params(df_train: pd.DataFrame) -> dict:
    """
    Compute all imputation statistics from the training set only.
    Call this before encoding; pass the returned dict to encode_features().
    """
    wgi_cols = list(WGI_SHEETS.values())
    params: dict = {}

    # Numeric medians
    params["log_investment_median"] = np.log10(
        df_train["investment"].clip(lower=0.01)
    ).median()
    params["period_median"]  = df_train["period"].median()
    params["private_median"] = df_train["private"].median()

    # WGI regional means + global fallback medians (train only)
    params["wgi_regional_means"]  = {}
    params["wgi_global_medians"]  = {}
    for col in wgi_cols:
        if col in df_train.columns:
            params["wgi_regional_means"][col] = (
                df_train.groupby("Region")[col].mean().to_dict()
            )
            params["wgi_global_medians"][col] = df_train[col].median()

    # Macro medians (fallback for any years outside hardcoded range)
    for col in ["energy_idx", "metals_idx", "treasury_10y"]:
        if col in df_train.columns:
            params[f"{col}_median"] = df_train[col].median()

    # Top ssectors (frequency-ranked on train)
    params["top_ssectors"] = (
        df_train["ssector"].value_counts().head(15).index.tolist()
    )

    # Label maps for the 3 interaction features — built on train combos only
    cam_sec  = (df_train["CAM"].fillna("MISSING") + "__" +
                df_train["sector"].fillna("MISSING"))
    sec_type = (df_train["sector"].fillna("MISSING") + "__" +
                df_train["type"].fillna("MISSING"))
    reg_coh  = (df_train["Region"].fillna("MISSING") + "__" +
                df_train["fcy_cohort"].astype(str))
    params["cam_x_sector_map"]   = _build_label_map(cam_sec)
    params["sect_x_type_map"]    = _build_label_map(sec_type)
    params["reg_x_coh_map"]      = _build_label_map(reg_coh)

    return params


def encode_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Encode features using imputation stats from `params` (pre-computed on train).
    Applies identically to train and test — no statistics computed here.
    """
    df = df.copy()

    # ── Numeric transforms ──────────────────────────────────────────────
    df["log_investment"]    = np.log10(df["investment"].clip(lower=0.01))
    df["missing_investment"] = df["investment"].isna().astype(int)
    df["log_investment"]    = df["log_investment"].fillna(params["log_investment_median"])

    df["missing_period"] = df["period"].isna().astype(int)
    df["period"]         = df["period"].fillna(params["period_median"])

    df["missing_private"] = df["private"].isna().astype(int)
    df["private"]         = df["private"].fillna(params["private_median"])

    # ── WGI: fill with train regional mean, then train global median ────
    wgi_cols = list(WGI_SHEETS.values())
    for col in wgi_cols:
        if col in df.columns:
            reg_means = params["wgi_regional_means"].get(col, {})
            df[col] = df[col].fillna(df["Region"].map(reg_means))
            df[col] = df[col].fillna(params["wgi_global_medians"].get(col, 0.0))
            df[f"missing_{col}"] = df[col].isna().astype(int)

    # ── Macro: fill with train median ───────────────────────────────────
    for col in ["energy_idx", "metals_idx", "treasury_10y"]:
        if col in df.columns:
            df[col] = df[col].fillna(params.get(f"{col}_median", 0.0))

    # ── Binary / flag features ───────────────────────────────────────────
    df["high_risk_sector"]   = df["sector"].isin(["ICT", "Energy"]).astype(int)
    df["is_greenfield"]      = (df["type"] == "Greenfield project").astype(int)
    df["is_ppp"]             = (df["PPP"] == "PPP Project").astype(int)
    df["is_ict"]             = (df["sector"] == "ICT").astype(int)
    df["has_govt_guarantee"] = (~df["GGC"].isin(["NA", "MISSING"])).astype(int)
    df["is_unsolicited"]     = (df["UP"] == "Yes").astype(int)
    df["is_ida_eligible"]    = (
        df["IDA"].fillna("").str.strip().str.lower() == "yes"
    ).astype(int)
    df = df.drop(columns=["UP", "PPP", "IDA"], errors="ignore")

    # ── Subsector: collapse using train-derived top-N list ───────────────
    top_ss = params["top_ssectors"]
    df["ssector_grouped"] = df["ssector"].apply(
        lambda x: x if x in top_ss else "Other"
    )

    # ── Interaction terms: 3 label-encoded integers (train maps applied) ─
    # Unseen test combos → 0
    cam_sec  = df["CAM"].fillna("MISSING") + "__" + df["sector"].fillna("MISSING")
    sec_type = df["sector"].fillna("MISSING") + "__" + df["type"].fillna("MISSING")
    reg_coh  = df["Region"].fillna("MISSING") + "__" + df["fcy_cohort"].astype(str)
    df["int_cam_x_sector"]  = cam_sec.map(params["cam_x_sector_map"]).fillna(0).astype(int)
    df["int_sect_x_type"]   = sec_type.map(params["sect_x_type_map"]).fillna(0).astype(int)
    df["int_reg_x_coh"]     = reg_coh.map(params["reg_x_coh_map"]).fillna(0).astype(int)

    # ── Fill categorical NaNs before one-hot encoding ────────────────────
    cat_cols = ["sector", "ssector_grouped", "type", "Region", "income",
                "CAM", "bid_crit", "GGC", "MLS", "BS", "fcy_cohort"]
    df = _fill_categorical_missing(df, cat_cols)

    # ── One-hot encoding ──────────────────────────────────────────────────
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — DROP NON-FEATURE COLUMNS AND SPLIT
# ═══════════════════════════════════════════════════════════════════════════

DROP_COLS = [
    "ID", "split", "status_n", "country", "name",
    "numberb", "pcapacity", "PRS", "technol",  # excluded per audit
    "investment", "private",                    # replaced by log_investment, private (numeric)
    "ssector",                                  # replaced by ssector_grouped one-hot
]


def build_features(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_train = PROC_DIR / "features_train.parquet"
    out_test  = PROC_DIR / "features_test.parquet"

    if out_train.exists() and out_test.exists() and not force:
        print("Features already built. Loading from cache.")
        return pd.read_parquet(out_train), pd.read_parquet(out_test)

    print("\n─── Step 1: Audit ────────────────────────────────────────────")
    print_audit()

    print("─── Step 2: Load data ────────────────────────────────────────")
    df = load_data()

    print("\n─── Step 3: FCY cohort ────────────────────────────────────────")
    df = add_fcy_cohort(df)

    print("─── Step 4: WGI enrichment ────────────────────────────────────")
    try:
        wgi = download_wgi()
        df = join_wgi(df, wgi)
    except Exception as e:
        print(f"  WGI enrichment FAILED ({e}) — continuing without WGI features")

    print("\n─── Step 5: Pink Sheet commodity prices ──────────────────────")
    try:
        ps = download_pinksheet()
        df = join_pinksheet(df, ps)
    except Exception as e:
        print(f"  Pink Sheet FAILED ({e}) — continuing without commodity features")

    print("\n─── Step 6: US 10Y Treasury ──────────────────────────────────")
    df = join_treasury(df)

    # ── Split BEFORE any encoding to prevent imputation leakage ─────────
    df_tr = df[df["FCY"] <= TRAIN_CUTOFF_YEAR].copy()
    df_te = df[df["FCY"] >= TEST_START_YEAR].copy()

    print("\n─── Step 7: Compute imputation params from TRAIN only ────────")
    params = compute_impute_params(df_tr)
    print("  Imputation stats computed on train only: YES")

    print("\n─── Step 8: Feature encoding ─────────────────────────────────")
    df_tr = encode_features(df_tr, params)
    df_te = encode_features(df_te, params)

    # Drop forbidden / admin / excluded columns
    for split in [df_tr, df_te]:
        drop_present = [c for c in DROP_COLS if c in split.columns]
        split.drop(columns=drop_present, inplace=True, errors="ignore")
        obj_cols = split.select_dtypes(include="object").columns.tolist()
        if obj_cols:
            print(f"  WARNING: dropping remaining object columns: {obj_cols}")
            split.drop(columns=obj_cols, inplace=True)

    # Align test columns to train (fill missing dummies with 0, drop unseen)
    train_feat_cols = [c for c in df_tr.columns if c not in ("FCY", "target")]
    df_te = df_te.reindex(columns=["FCY", "target"] + train_feat_cols, fill_value=0)

    train = df_tr.reset_index(drop=True)
    test  = df_te.reset_index(drop=True)

    print("\n─── Step 9: Save ─────────────────────────────────────────────")
    train.to_parquet(out_train, index=False)
    test.to_parquet(out_test,  index=False)

    print(f"  features_train.parquet: {train.shape}")
    print(f"  features_test.parquet:  {test.shape}")

    return train, test


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════

def summary_report(train: pd.DataFrame, test: pd.DataFrame) -> None:
    print("\n" + "═" * 70)
    print("  FEATURE ENGINEERING SUMMARY")
    print("═" * 70)

    feature_cols = [c for c in train.columns if c not in ("FCY", "target")]
    print(f"\n  Total features: {len(feature_cols)}")

    # Missingness per feature (train set)
    miss = (train[feature_cols].isna().mean() * 100).sort_values(ascending=False)
    miss_nonzero = miss[miss > 0]
    if len(miss_nonzero):
        print(f"\n  Features with remaining missingness (train):")
        for feat, pct in miss_nonzero.items():
            print(f"    {feat:45s}  {pct:.1f}%")
    else:
        print("\n  No missing values in feature matrix (all imputed).")

    # Class balance
    print(f"\n  Class balance:")
    print(f"    TRAIN (FCY ≤ {TRAIN_CUTOFF_YEAR}): "
          f"n={len(train):,}  "
          f"distress={train['target'].sum():,} ({train['target'].mean()*100:.1f}%)")
    print(f"    TEST  (FCY {TEST_START_YEAR}-{EXPOSURE_CUTOFF}): "
          f"n={len(test):,}  "
          f"distress={test['target'].sum():,} ({test['target'].mean()*100:.1f}%)")

    train_rate = train["target"].mean()
    test_rate  = test["target"].mean()
    if test_rate < train_rate * 0.3:
        print(f"\n  ⚠  WARNING: Test-set distress rate ({test_rate*100:.1f}%) is very low "
              f"vs. train ({train_rate*100:.1f}%). This reflects the 2015-2017 "
              f"reporting-lag issue identified in EDA. Evaluate model on test-set "
              f"precision/recall carefully — strong AUC may partly reflect the "
              f"near-zero base rate, not genuine lift.")

    print(f"\n  Feature groups:")
    wgi_feats   = [c for c in feature_cols if "wgi_" in c]
    macro_feats = [c for c in feature_cols if c in ("energy_idx","metals_idx","treasury_10y")]
    cat_feats   = [c for c in feature_cols if c.startswith(("sector_","type_","Region_",
                   "income_","CAM_","bid_crit_","GGC_","fcy_cohort_","ssector_",
                   "MLS_","BS_"))]
    interact    = [c for c in feature_cols if c.startswith("int_")]
    numeric     = [c for c in feature_cols if c not in wgi_feats + macro_feats + cat_feats + interact]
    print(f"    Categorical one-hot:   {len(cat_feats):3d}")
    print(f"    Interaction (label-enc):{len(interact):3d}")
    print(f"    WGI governance:        {len(wgi_feats):3d}")
    print(f"    Macro (commodity/rate): {len(macro_feats):3d}")
    print(f"    Numeric / binary:      {len(numeric):3d}")
    print("═" * 70)


if __name__ == "__main__":
    train, test = build_features(force=True)
    summary_report(train, test)
