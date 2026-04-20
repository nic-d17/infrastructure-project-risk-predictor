"""
Prediction pipeline: form inputs → 87 features → calibrated P(distress) + SHAP.
Self-contained — no dependency on notebooks/features_script.py.
"""

import json
import pickle
import io
from pathlib import Path
from functools import lru_cache

import joblib

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent.parent

WGI_COLS = [
    "wgi_voice_accountability",
    "wgi_political_stability",
    "wgi_gov_effectiveness",
    "wgi_regulatory_quality",
    "wgi_rule_of_law",
    "wgi_control_corruption",
]

US_10Y_ANNUAL = {
    1990: 8.55, 1991: 7.86, 1992: 7.01, 1993: 5.87, 1994: 7.09,
    1995: 6.57, 1996: 6.44, 1997: 6.35, 1998: 5.26, 1999: 5.64,
    2000: 6.03, 2001: 5.02, 2002: 4.61, 2003: 4.02, 2004: 4.27,
    2005: 4.29, 2006: 4.79, 2007: 4.63, 2008: 3.66, 2009: 3.26,
    2010: 3.22, 2011: 2.78, 2012: 1.80, 2013: 2.35, 2014: 2.54,
    2015: 2.14, 2016: 1.84, 2017: 2.33, 2018: 2.91, 2019: 2.14,
    2020: 0.89, 2021: 1.45, 2022: 2.95, 2023: 3.96, 2024: 4.20,
    2025: 4.30, 2026: 4.30,
}

# Sector → valid subsectors in this model
SECTOR_SSECTORS = {
    "Energy":                ["Electricity", "Natural Gas", "Other"],
    "ICT":                   ["Telecom", "Other"],
    "Transport":             ["Airports", "Railroads", "Seaports", "Toll Roads", "Other"],
    "Water and sewerage":    ["Water Utility", "Treatment plant", "Treatment/ Disposal", "Other"],
    "Municipal Solid Waste": ["Collection and Transport", "Integrated MSW", "Other"],
}

REGION_LABELS = {
    "AFR": "Sub-Saharan Africa",
    "EAP": "East Asia & Pacific",
    "ECA": "Europe & Central Asia",
    "LAC": "Latin America & Caribbean",
    "MENA": "Middle East & North Africa",
    "SAR": "South Asia",
}

EXAMPLE_PROJECTS = {
    "High-risk (Argentina Energy, 1992)": {
        "country": "Argentina",
        "Region": "LAC",
        "income": "Upper middle income",
        "FCY": 1992,
        "sector": "Energy",
        "ssector": "Electricity",
        "type": "Brownfield",
        "CAM": "Competitive bidding",
        "bid_crit": "Highest price paid to government",
        "GGC": "Federal",
        "MLS": "With MLS",
        "BS": "Without BLS",
        "investment": 428.0,
        "period": 95.0,
        "is_ppp": True,
        "is_unsolicited": False,
        "is_ida_eligible": False,
        "bordercountries": 0,
    },
    "Low-risk (Russia Energy, 1993)": {
        "country": "Russian Federation",
        "Region": "ECA",
        "income": "Upper middle income",
        "FCY": 1993,
        "sector": "Energy",
        "ssector": "Electricity",
        "type": "Divestiture",
        "CAM": "MISSING",
        "bid_crit": "Not Applicable",
        "GGC": "MISSING",
        "MLS": "Without MLS",
        "BS": "Without BLS",
        "investment": np.nan,
        "period": np.nan,
        "is_ppp": False,
        "is_unsolicited": False,
        "is_ida_eligible": False,
        "bordercountries": 0,
    },
    "Boundary (Russia ICT, 1994)": {
        "country": "Russian Federation",
        "Region": "ECA",
        "income": "Upper middle income",
        "FCY": 1994,
        "sector": "ICT",
        "ssector": "Telecom",
        "type": "Divestiture",
        "CAM": "MISSING",
        "bid_crit": "Not Applicable",
        "GGC": "MISSING",
        "MLS": "Without MLS",
        "BS": "Without BLS",
        "investment": np.nan,
        "period": np.nan,
        "is_ppp": False,
        "is_unsolicited": False,
        "is_ida_eligible": False,
        "bordercountries": 0,
    },
}


def assign_category(feat: str) -> str:
    if feat.startswith("wgi_") or feat.startswith("missing_wgi_"):
        return "Governance (WGI)"
    if feat in ("treasury_10y", "energy_idx", "metals_idx"):
        return "Macro"
    if feat.startswith("fcy_cohort_"):
        return "Cohort (temporal)"
    if feat in ("log_investment", "missing_investment", "private", "missing_private"):
        return "Size / Investment"
    if feat.startswith("sector_") or feat.startswith("ssector_") or feat in ("high_risk_sector", "is_ict"):
        return "Sector"
    if (feat.startswith("type_") or feat.startswith("MLS_") or feat.startswith("BS_") or
            feat in ("is_greenfield", "is_ppp", "period", "missing_period", "int_sect_x_type")):
        return "Project Type / Structure"
    if (feat.startswith("CAM_") or feat.startswith("bid_crit_") or feat.startswith("GGC_") or
            feat in ("has_govt_guarantee", "is_unsolicited", "int_cam_x_sector")):
        return "Contract / Procurement"
    if (feat.startswith("Region_") or feat.startswith("income_") or
            feat in ("is_ida_eligible", "bordercountries", "int_reg_x_coh")):
        return "Geography"
    return "Other"


@lru_cache(maxsize=1)
def load_artifacts() -> dict:
    rf  = joblib.load(ROOT / "models" / "best_model_uncalibrated.pkl")
    cal = joblib.load(ROOT / "models" / "best_model_calibrated.pkl")
    with open(ROOT / "models" / "feature_cols.json") as f:
        feature_cols = json.load(f)
    with open(ROOT / "models" / "impute_params.pkl", "rb") as f:
        params = pickle.load(f)

    country_lookup = pd.read_csv(ROOT / "models" / "country_lookup.csv")
    wgi_lookup = pd.read_csv(ROOT / "models" / "wgi_by_country.csv")
    macro_lookup = pd.read_csv(ROOT / "models" / "macro_lookup.csv")

    explainer = shap.TreeExplainer(rf)
    ev = explainer.expected_value
    base_value = float(ev[1]) if hasattr(ev, "__len__") else float(ev)

    return {
        "rf": rf,
        "cal": cal,
        "feature_cols": feature_cols,
        "params": params,
        "country_lookup": country_lookup,
        "wgi_lookup": wgi_lookup,
        "macro_lookup": macro_lookup,
        "explainer": explainer,
        "base_value": base_value,
    }


def _fcy_cohort(fcy: int) -> str:
    if fcy < 2000:
        return "pre-2000"
    elif fcy < 2008:
        return "2000-2007"
    elif fcy < 2015:
        return "2008-2014"
    else:
        return "2015-2017"


def _get_wgi(country: str, fcy: int, wgi_lookup: pd.DataFrame, params: dict) -> dict:
    """Return WGI dict for country at FCY year (clamped to 1996)."""
    wgi_year = max(fcy, 1996)
    subset = wgi_lookup[wgi_lookup["country"] == country]
    if subset.empty:
        return {c: params["wgi_global_medians"].get(c, 0.0) for c in WGI_COLS}

    # Find closest year
    years = subset["year"].values
    closest = years[np.argmin(np.abs(years - wgi_year))]
    row = subset[subset["year"] == closest].iloc[0]

    result = {}
    for col in WGI_COLS:
        val = row.get(col, np.nan)
        if pd.isna(val):
            val = params["wgi_regional_means"].get(col, {}).get(
                _get_region_for_country(country), params["wgi_global_medians"].get(col, 0.0)
            )
        result[col] = float(val)
    return result


def _get_region_for_country(country: str) -> str:
    arts = load_artifacts()
    match = arts["country_lookup"][arts["country_lookup"]["country"] == country]
    if not match.empty:
        return match.iloc[0]["Region"]
    return "LAC"


def _get_macro(fcy: int, macro_lookup: pd.DataFrame, params: dict) -> dict:
    row = macro_lookup[macro_lookup["year"] == fcy]
    if row.empty:
        # Use most recent available
        row = macro_lookup[macro_lookup["year"] <= fcy].tail(1)
    if row.empty:
        return {
            "energy_idx": params.get("energy_idx_median", 20.0),
            "metals_idx": params.get("metals_idx_median", 50.0),
            "treasury_10y": params.get("treasury_10y_median", 4.5),
        }
    r = row.iloc[0]
    treasury = US_10Y_ANNUAL.get(fcy, US_10Y_ANNUAL.get(max(k for k in US_10Y_ANNUAL if k <= fcy), 4.20))
    return {
        "energy_idx": float(r["energy_idx"]) if not pd.isna(r.get("energy_idx", np.nan)) else params.get("energy_idx_median", 20.0),
        "metals_idx": float(r["metals_idx"]) if not pd.isna(r.get("metals_idx", np.nan)) else params.get("metals_idx_median", 50.0),
        "treasury_10y": treasury,
    }


def build_feature_row(inputs: dict) -> np.ndarray:
    """
    Encode a single project input dict into the 87-feature vector.
    inputs keys: country, Region, income, FCY, sector, ssector, type,
                 CAM, bid_crit, GGC, MLS, BS, investment, period,
                 is_ppp, is_unsolicited, is_ida_eligible, bordercountries
    """
    arts = load_artifacts()
    params = arts["params"]
    feature_cols = arts["feature_cols"]
    wgi_lookup = arts["wgi_lookup"]
    macro_lookup = arts["macro_lookup"]

    fcy = int(inputs["FCY"])
    cohort = _fcy_cohort(fcy)
    wgi = _get_wgi(inputs["country"], fcy, wgi_lookup, params)
    macro = _get_macro(fcy, macro_lookup, params)

    inv = inputs.get("investment", np.nan)
    if pd.isna(inv) or inv <= 0:
        log_inv = params["log_investment_median"]
        miss_inv = 1
    else:
        log_inv = float(np.log10(float(inv)))
        miss_inv = 0

    period = inputs.get("period", np.nan)
    if pd.isna(period):
        period_val = params["period_median"]
        miss_period = 1
    else:
        period_val = float(period)
        miss_period = 0

    sector  = inputs.get("sector", "Energy")
    ptype   = inputs.get("type", "Greenfield project")
    cam     = inputs.get("CAM", "MISSING")
    region  = inputs.get("Region", "LAC")
    income  = inputs.get("income", "Upper middle income")
    ssector = inputs.get("ssector", "Other")
    bid_crit = inputs.get("bid_crit", "Not Available")
    ggc     = inputs.get("GGC", "MISSING")
    mls     = inputs.get("MLS", "Without MLS")
    bs      = inputs.get("BS", "Without BLS")

    # Map ssector to grouped (same top-15 logic)
    top_ss = params["top_ssectors"]
    ssector_grouped = ssector if ssector in top_ss else "Other"

    # Interactions
    cam_sec  = f"{cam}__{sector}"
    sec_type = f"{sector}__{ptype}"
    reg_coh  = f"{region}__{cohort}"
    int_cam_x_sector = params["cam_x_sector_map"].get(cam_sec, 0)
    int_sect_x_type  = params["sect_x_type_map"].get(sec_type, 0)
    int_reg_x_coh    = params["reg_x_coh_map"].get(reg_coh, 0)

    # WGI missingness indicators
    miss_wgi = {f"missing_{c}": (0 if c in wgi and not pd.isna(wgi[c]) else 1) for c in WGI_COLS}

    # Build flat dict of all features
    feat = {
        "bordercountries": int(inputs.get("bordercountries", 0)),
        "period": period_val,
        "log_investment": log_inv,
        "missing_investment": miss_inv,
        "missing_period": miss_period,
        "missing_private": 0,
        "energy_idx": macro["energy_idx"],
        "metals_idx": macro["metals_idx"],
        "treasury_10y": macro["treasury_10y"],
        "high_risk_sector": int(sector in ("ICT", "Energy")),
        "is_greenfield": int(ptype == "Greenfield project"),
        "is_ppp": int(bool(inputs.get("is_ppp", False))),
        "is_ict": int(sector == "ICT"),
        "has_govt_guarantee": int(ggc not in ("NA", "MISSING")),
        "is_unsolicited": int(bool(inputs.get("is_unsolicited", False))),
        "is_ida_eligible": int(bool(inputs.get("is_ida_eligible", False))),
        "int_cam_x_sector": int_cam_x_sector,
        "int_sect_x_type": int_sect_x_type,
        "int_reg_x_coh": int_reg_x_coh,
    }
    for c in WGI_COLS:
        feat[c] = wgi[c]
    feat.update(miss_wgi)

    # One-hot: sector
    for s in ["Energy", "ICT", "Municipal Solid Waste", "Transport", "Water and sewerage"]:
        feat[f"sector_{s}"] = int(sector == s)

    # One-hot: ssector_grouped
    for ss in ["Airports", "Collection and Transport", "Electricity", "Integrated MSW",
               "Natural Gas", "Railroads", "Seaports", "Telecom", "Toll Roads",
               "Treatment plant", "Treatment/ Disposal", "Water Utility"]:
        feat[f"ssector_grouped_{ss}"] = int(ssector_grouped == ss)

    # One-hot: type
    for t in ["Brownfield", "Divestiture", "Greenfield project", "Management and lease contract"]:
        feat[f"type_{t}"] = int(ptype == t)

    # One-hot: Region
    for r in ["AFR", "EAP", "ECA", "LAC", "MENA", "SAR"]:
        feat[f"Region_{r}"] = int(region == r)

    # One-hot: income
    for i in ["Low income", "Lower middle income", "Upper middle income"]:
        feat[f"income_{i}"] = int(income == i)

    # One-hot: CAM
    for c in ["Competitive bidding", "Competitive negotiation", "Direct negotiation", "License scheme", "MISSING"]:
        feat[f"CAM_{c}"] = int(cam == c)

    # One-hot: bid_crit
    for b in ["Highest % of revenue share with government", "Highest new investment",
              "Highest price paid to government", "Lowest cost of construction or operation",
              "Lowest government payments", "Lowest subsidy required", "Lowest tariff",
              "Not Applicable", "Not Available", "Other"]:
        feat[f"bid_crit_{b}"] = int(bid_crit == b)

    # One-hot: GGC
    for g in ["Federal", "Local", "MISSING", "State/Provincial"]:
        feat[f"GGC_{g}"] = int(ggc == g)

    # One-hot: MLS
    for m in ["With MLS", "Without MLS"]:
        feat[f"MLS_{m}"] = int(mls == m)

    # One-hot: BS
    for b in ["With BLS", "Without BLS"]:
        feat[f"BS_{b}"] = int(bs == b)

    # One-hot: fcy_cohort
    for coh in ["pre-2000", "2000-2007", "2008-2014"]:
        feat[f"fcy_cohort_{coh}"] = int(cohort == coh)

    # Align to exact feature_cols order
    row = np.array([feat.get(col, 0.0) for col in feature_cols], dtype="float32")
    return row


def predict(inputs: dict) -> dict:
    """
    Returns dict with keys:
      calibrated_prob, shap_values (1d array), feature_cols, base_value
    """
    arts = load_artifacts()
    row = build_feature_row(inputs)
    X = row.reshape(1, -1)

    calibrated_prob = float(arts["cal"].predict_proba(X)[0, 1])

    sv_raw = arts["explainer"].shap_values(X)
    if isinstance(sv_raw, list):
        shap_vals = sv_raw[1][0]
    elif sv_raw.ndim == 3:
        shap_vals = sv_raw[0, :, 1]
    else:
        shap_vals = sv_raw[0]

    return {
        "calibrated_prob": calibrated_prob,
        "shap_values": shap_vals,
        "feature_cols": arts["feature_cols"],
        "base_value": arts["base_value"],
        "row": row,
    }


def risk_bucket(p: float) -> tuple[str, str]:
    """Returns (label, hex_color)."""
    if p < 0.05:
        return "Low", "#2ecc71"
    elif p < 0.15:
        return "Moderate", "#f39c12"
    elif p < 0.30:
        return "Elevated", "#e67e22"
    else:
        return "High", "#e74c3c"


def waterfall_figure(result: dict, max_display: int = 10) -> bytes:
    """Return PNG bytes of a SHAP waterfall plot for one prediction."""
    exp = shap.Explanation(
        values=result["shap_values"],
        base_values=result["base_value"],
        data=result["row"],
        feature_names=result["feature_cols"],
    )
    plt.figure(figsize=(9, 5))
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.read()


def category_figure(result: dict) -> bytes:
    """Return PNG bytes of a horizontal bar chart of category-level SHAP contributions."""
    shap_vals = result["shap_values"]
    feature_cols = result["feature_cols"]

    cats = [assign_category(f) for f in feature_cols]
    cat_series = pd.Series(np.abs(shap_vals), index=cats)
    cat_sum = cat_series.groupby(cat_series.index).sum().sort_values(ascending=True)

    colors = {
        "Governance (WGI)": "#3498db",
        "Project Type / Structure": "#9b59b6",
        "Macro": "#e74c3c",
        "Contract / Procurement": "#e67e22",
        "Size / Investment": "#2ecc71",
        "Sector": "#1abc9c",
        "Geography": "#95a5a6",
        "Cohort (temporal)": "#bdc3c7",
        "Other": "#ecf0f1",
    }
    bar_colors = [colors.get(c, "#95a5a6") for c in cat_sum.index]
    total = cat_sum.sum()
    pcts = (cat_sum / total * 100).round(1)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.barh(cat_sum.index, cat_sum.values, color=bar_colors, height=0.6)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + total * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{pct}%", va="center", fontsize=8)
    ax.set_xlabel("Sum |SHAP value|", fontsize=9)
    ax.set_title("Prediction drivers by category", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.read()
