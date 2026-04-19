"""Infrastructure Project Risk Predictor — Streamlit app."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.app.predictor import (
    load_artifacts,
    predict,
    risk_bucket,
    waterfall_figure,
    category_figure,
    SECTOR_SSECTORS,
    REGION_LABELS,
    EXAMPLE_PROJECTS,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Infrastructure Project Risk Predictor",
    page_icon="🏗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .prob-card {
        border-radius: 10px;
        padding: 24px 32px;
        margin-bottom: 16px;
        text-align: center;
    }
    .prob-number { font-size: 3.2rem; font-weight: 700; line-height: 1; }
    .prob-label  { font-size: 1.1rem; font-weight: 600; margin-top: 6px; }
    .disclaimer  { font-size: 0.75rem; color: #888; margin-top: 12px; }
    .section-header { font-size: 0.85rem; font-weight: 600; text-transform: uppercase;
                       letter-spacing: 0.06em; color: #666; margin-bottom: 4px; }
    div[data-testid="stExpander"] > div { padding: 0 0 8px 0; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar: methodology ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Methodology")
    st.markdown("""
**Model:** Random Forest (300 trees, class-weight balanced), calibrated with Platt scaling.
Trained on 8,435 World Bank PPI infrastructure projects (financial close ≤ 2014) using
5-fold stratified cross-validation. CV AUC: **0.823**, Brier: **0.049**, PR-AUC: **0.294**.

**Target:** Binary distress/cancellation within 7 years of financial close. Projects that
became distressed have, by definition, experienced a failure of the underwriting assumptions —
this is the question a credit committee is actually answering when underwriting infrastructure debt.

**Features:** 87 features across 8 categories — WGI governance scores (23.9% of predictive
signal), project structure (18.1%), macro environment at financial close (16.0%), and
contract/procurement structure (12.8%). SHAP leakage audit confirmed all top-20 features
are observable at financial close year. Full methodology in the project plan.

**Temporal discipline:** Train/test split is temporal (not random) — mimics real underwriting
where you train on history and score future deals. Test set (FCY 2015–2017) has a near-zero
distress rate reflecting the PPI reporting lag, not genuine absence of risk.
    """)
    st.markdown("---")
    st.markdown(
        "[GitHub Repo](https://github.com/nicholasdaal/cost-overrun-predictor) · "
        "[Project Plan](https://github.com/nicholasdaal/cost-overrun-predictor/blob/main/docs/Cost_Overrun_Predictor_Project_Plan.md)"
    )

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# Infrastructure Project Risk Predictor")
st.markdown(
    "Estimates probability of project distress within 7 years of financial close, "
    "trained on 11,640 World Bank PPI infrastructure projects across 137 countries (1990–2024)."
)

meth_link = st.button("Methodology & model details", key="meth_btn")
if meth_link:
    st.info("Open the sidebar (top-left ▶) for the full methodology explainer.", icon="ℹ️")

st.markdown("---")

# ── Pre-load artifacts (cached) ────────────────────────────────────────────────
with st.spinner("Loading model..."):
    arts = load_artifacts()
country_list = sorted(arts["country_lookup"]["country"].dropna().unique().tolist())

# ── Session state for form values ─────────────────────────────────────────────
if "form" not in st.session_state:
    st.session_state.form = {
        "country": "Brazil",
        "FCY": 2010,
        "sector": "Energy",
        "ssector": "Electricity",
        "type": "Greenfield project",
        "investment": 500.0,
        "period": 25.0,
        "CAM": "Competitive bidding",
        "bid_crit": "Lowest tariff",
        "GGC": "Federal",
        "MLS": "Without MLS",
        "BS": "Without BLS",
        "is_ppp": True,
        "is_unsolicited": False,
        "is_ida_eligible": False,
        "bordercountries": 0,
    }

# ── Example project buttons ────────────────────────────────────────────────────
st.markdown('<p class="section-header">Load example project</p>', unsafe_allow_html=True)
ex_cols = st.columns(3)
for i, (name, vals) in enumerate(EXAMPLE_PROJECTS.items()):
    if ex_cols[i].button(name, use_container_width=True):
        st.session_state.form = dict(vals)
        st.rerun()

st.markdown("")

# ── Main layout: 60 / 40 ──────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
# LEFT COLUMN — Input form
# ─────────────────────────────────────────────────────────────────────────────
with left:
    st.markdown("### Project inputs")

    def _form_val(key, default=None):
        return st.session_state.form.get(key, default)

    # ── Project basics ────────────────────────────────────────────────────────
    with st.expander("Project basics", expanded=True):
        b_col1, b_col2 = st.columns(2)

        with b_col1:
            sector = st.selectbox(
                "Sector",
                list(SECTOR_SSECTORS.keys()),
                index=list(SECTOR_SSECTORS.keys()).index(_form_val("sector", "Energy")),
                key="sector_sel",
            )

        valid_ss = SECTOR_SSECTORS[sector]
        prev_ss = _form_val("ssector", valid_ss[0])
        ss_default = prev_ss if prev_ss in valid_ss else valid_ss[0]

        with b_col2:
            ssector = st.selectbox(
                "Subsector",
                valid_ss,
                index=valid_ss.index(ss_default),
                key="ssector_sel",
            )

        b_col3, b_col4 = st.columns(2)
        type_options = ["Greenfield project", "Brownfield", "Divestiture", "Management and lease contract"]
        with b_col3:
            ptype = st.selectbox(
                "Project type",
                type_options,
                index=type_options.index(_form_val("type", "Greenfield project")),
                key="type_sel",
            )

        with b_col4:
            investment = st.number_input(
                "Investment size (USD millions)",
                min_value=0.0,
                max_value=100_000.0,
                value=float(_form_val("investment", 500.0)) if not pd.isna(_form_val("investment", 500.0)) else 0.0,
                step=10.0,
                key="investment_sel",
            )

        b_col5, b_col6 = st.columns(2)
        with b_col5:
            country = st.selectbox(
                "Country",
                country_list,
                index=country_list.index(_form_val("country", "Brazil")) if _form_val("country", "Brazil") in country_list else 0,
                key="country_sel",
            )

        # Auto-populate Region + income from country lookup
        cl = arts["country_lookup"]
        match = cl[cl["country"] == country]
        auto_region = match.iloc[0]["Region"] if not match.empty else "LAC"
        auto_income = match.iloc[0]["income"] if not match.empty else "Upper middle income"

        with b_col6:
            fcy = st.number_input(
                "Financial close year (FCY)",
                min_value=1990,
                max_value=2026,
                value=int(_form_val("FCY", 2010)),
                step=1,
                key="fcy_sel",
            )

        st.caption(
            f"Auto-populated from country: Region = **{REGION_LABELS.get(auto_region, auto_region)}** "
            f"({auto_region}) · Income = **{auto_income}**"
        )

    # ── Deal structure ────────────────────────────────────────────────────────
    with st.expander("Deal structure", expanded=True):
        d_col1, d_col2 = st.columns(2)

        cam_options = ["Competitive bidding", "Competitive negotiation", "Direct negotiation", "License scheme", "MISSING"]
        with d_col1:
            cam = st.selectbox(
                "Contract award method (CAM)",
                cam_options,
                index=cam_options.index(_form_val("CAM", "Competitive bidding")),
                key="cam_sel",
            )

        bid_options = [
            "Lowest tariff", "Lowest government payments", "Lowest cost of construction or operation",
            "Lowest subsidy required", "Highest price paid to government",
            "Highest % of revenue share with government", "Highest new investment",
            "Not Applicable", "Not Available", "Other",
        ]
        with d_col2:
            bid_crit = st.selectbox(
                "Bidding criterion",
                bid_options,
                index=bid_options.index(_form_val("bid_crit", "Lowest tariff")) if _form_val("bid_crit", "Lowest tariff") in bid_options else 0,
                key="bid_sel",
            )

        d_col3, d_col4 = st.columns(2)
        ggc_options = ["Federal", "State/Provincial", "Local", "MISSING"]
        with d_col3:
            ggc = st.selectbox(
                "Government guarantee structure (GGC)",
                ggc_options,
                index=ggc_options.index(_form_val("GGC", "Federal")) if _form_val("GGC", "Federal") in ggc_options else 3,
                key="ggc_sel",
            )

        mls_options = ["With MLS", "Without MLS"]
        with d_col4:
            mls = st.selectbox(
                "Multilateral lending support",
                mls_options,
                index=mls_options.index(_form_val("MLS", "Without MLS")),
                key="mls_sel",
            )

        d_col5, d_col6 = st.columns(2)
        bs_options = ["With BLS", "Without BLS"]
        with d_col5:
            bs = st.selectbox(
                "Bond structure",
                bs_options,
                index=bs_options.index(_form_val("BS", "Without BLS")),
                key="bs_sel",
            )

        with d_col6:
            period = st.number_input(
                "Concession / construction period (years)",
                min_value=0.0,
                max_value=99.0,
                value=float(_form_val("period", 25.0)) if not pd.isna(_form_val("period", 25.0) or 0) else 0.0,
                step=1.0,
                key="period_sel",
            )

        d_col7, d_col8, d_col9 = st.columns(3)
        with d_col7:
            is_ppp = st.checkbox("PPP Project", value=bool(_form_val("is_ppp", True)), key="ppp_sel")
        with d_col8:
            is_unsolicited = st.checkbox("Unsolicited proposal", value=bool(_form_val("is_unsolicited", False)), key="up_sel")
        with d_col9:
            is_ida_eligible = st.checkbox("IDA-eligible country", value=bool(_form_val("is_ida_eligible", False)), key="ida_sel")

    # ── Context ──────────────────────────────────────────────────────────────
    with st.expander("Context at financial close", expanded=False):
        # Show macro values auto-populated from FCY
        macro_lookup = arts["macro_lookup"]
        macro_row = macro_lookup[macro_lookup["year"] == fcy]
        if macro_row.empty:
            macro_row = macro_lookup[macro_lookup["year"] <= fcy].tail(1)

        from src.app.predictor import US_10Y_ANNUAL
        treasury = US_10Y_ANNUAL.get(fcy, US_10Y_ANNUAL.get(max((k for k in US_10Y_ANNUAL if k <= fcy), default=1990), 4.20))

        if not macro_row.empty:
            r = macro_row.iloc[0]
            energy_disp = f"{r['energy_idx']:.1f}" if not pd.isna(r.get("energy_idx", np.nan)) else "N/A"
            metals_disp = f"{r['metals_idx']:.1f}" if not pd.isna(r.get("metals_idx", np.nan)) else "N/A"
        else:
            energy_disp = "N/A"
            metals_disp = "N/A"

        st.caption(
            f"**FCY {fcy} macro context (auto-populated, read-only):**  "
            f"US 10Y Treasury = **{treasury:.2f}%** · "
            f"Energy index = **{energy_disp}** · "
            f"Metals index = **{metals_disp}**"
        )
        st.caption(
            "These values are drawn from FRED (Treasury) and World Bank Pink Sheet (commodities) "
            "as of the financial close year. They are the macro environment the project was underwritten into."
        )

        # WGI preview
        from src.app.predictor import _get_wgi, WGI_COLS, _fcy_cohort
        wgi_vals = _get_wgi(country, fcy, arts["wgi_lookup"], arts["params"])
        wgi_df = pd.DataFrame({
            "Indicator": [c.replace("wgi_", "").replace("_", " ").title() for c in WGI_COLS],
            "Score": [f"{wgi_vals[c]:.3f}" for c in WGI_COLS],
        }).set_index("Indicator")
        st.caption(f"**WGI governance scores for {country} ({max(fcy, 1996)})**")
        st.dataframe(wgi_df, use_container_width=False)

    # ── Predict button ────────────────────────────────────────────────────────
    st.markdown("")
    predict_clicked = st.button("Predict distress probability", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT COLUMN — Output panel
# ─────────────────────────────────────────────────────────────────────────────
with right:
    st.markdown("### Prediction")

    if predict_clicked or "last_result" in st.session_state:
        if predict_clicked:
            # Assemble inputs from current widget state
            inputs = {
                "country":        country,
                "Region":         auto_region,
                "income":         auto_income,
                "FCY":            fcy,
                "sector":         sector,
                "ssector":        ssector,
                "type":           ptype,
                "investment":     investment if investment > 0 else np.nan,
                "period":         period if period > 0 else np.nan,
                "CAM":            cam,
                "bid_crit":       bid_crit,
                "GGC":            ggc,
                "MLS":            mls,
                "BS":             bs,
                "is_ppp":         is_ppp,
                "is_unsolicited": is_unsolicited,
                "is_ida_eligible": is_ida_eligible,
                "bordercountries": 0,
            }
            with st.spinner("Computing prediction..."):
                result = predict(inputs)
            st.session_state.last_result = result
            st.session_state.last_inputs = inputs
        else:
            result = st.session_state.last_result

        prob = result["calibrated_prob"]
        label, color = risk_bucket(prob)

        # ── Probability card ──────────────────────────────────────────────
        st.markdown(
            f"""<div class="prob-card" style="background:{color}22; border: 2px solid {color};">
            <div class="prob-number" style="color:{color}">{prob*100:.1f}%</div>
            <div class="prob-label" style="color:{color}">{label} risk</div>
            </div>""",
            unsafe_allow_html=True,
        )

        risk_desc = {
            "Low":      "P(distress) < 5% — well below training base rate of 7.2%.",
            "Moderate": "P(distress) 5–15% — near the training base rate.",
            "Elevated": "P(distress) 15–30% — 2–4× the average project in this dataset.",
            "High":     "P(distress) > 30% — materially above average; flag for deep diligence.",
        }
        st.caption(risk_desc[label])

        # ── SHAP waterfall ────────────────────────────────────────────────
        st.markdown("**Top drivers (SHAP waterfall)**")
        with st.spinner("Building SHAP waterfall..."):
            wf_png = waterfall_figure(result, max_display=10)
        st.image(wf_png, use_container_width=True)
        st.caption(
            "Each bar shows how much a feature pushed the prediction above (red) or "
            "below (blue) the model base rate. Features ranked by absolute contribution."
        )

        # ── Category breakdown ────────────────────────────────────────────
        st.markdown("**Prediction drivers by category**")
        with st.spinner("Building category chart..."):
            cat_png = category_figure(result)
        st.image(cat_png, use_container_width=True)

        # ── Disclaimer ────────────────────────────────────────────────────
        st.markdown(
            '<p class="disclaimer">This is a research tool. Probabilities are estimated on PPI '
            "distress/cancellation events within 7 years of financial close. "
            "Not investment advice.</p>",
            unsafe_allow_html=True,
        )

    else:
        st.info(
            "Fill in the project inputs on the left and click **Predict distress probability**. "
            "Or load one of the example projects above.",
            icon="👆",
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Infrastructure Project Risk Predictor · Nicholas Daal (Stanford MS Structural Engineering) · "
    "[GitHub](https://github.com/nicholasdaal/cost-overrun-predictor) · "
    "[Project Plan](https://github.com/nicholasdaal/cost-overrun-predictor/blob/main/docs/Cost_Overrun_Predictor_Project_Plan.md) · "
    "Built with Streamlit + scikit-learn + SHAP"
)
