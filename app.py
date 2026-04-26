"""
ECON 3916 Final Project — California Housing Predictor
Streamlit dashboard for a real estate investor benchmarking listings
against a Random Forest model trained on 1990 Census block-group features.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="California Housing Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Load model artifacts (cached)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base, "artifacts", "rf_model.pkl"))
    meta = joblib.load(os.path.join(base, "artifacts", "model_metadata.pkl"))
    return model, meta


model, meta = load_model()
RESIDUAL_STD = meta["residual_std"]
FEATURE_NAMES = meta["feature_names"]
FMEAN = meta["feature_means"]
FMIN = meta["feature_mins"]
FMAX = meta["feature_maxs"]

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🏡 California Housing Predictor")
st.markdown(
    "**Stakeholder:** A local California real estate investor screening listings. "
    "**Decision:** Is a given listing priced meaningfully below or above what a "
    "block group's demographic and housing fundamentals would predict?"
)
st.caption(
    "Model: Random Forest (200 trees, random_state=42) trained on the 1990 California "
    "Housing dataset. Predictions are **predictive benchmarks, not causal estimates** — "
    "they tell you what the price typically is for a block with these characteristics, "
    "not what would happen if you exogenously changed any single feature."
)

# ----------------------------------------------------------------------
# Sidebar: input controls
# ----------------------------------------------------------------------
st.sidebar.header("Block-Group Characteristics")
st.sidebar.caption("Adjust the sliders to describe the block group you are evaluating.")

def slider(label, key, fmt=None, step=None):
    lo = float(FMIN[key]); hi = float(FMAX[key]); default = float(FMEAN[key])
    kwargs = {"min_value": lo, "max_value": hi, "value": default}
    if step is not None:
        kwargs["step"] = step
    if fmt:
        kwargs["format"] = fmt
    return st.sidebar.slider(label, **kwargs)

med_inc   = slider("Median income (tens of $1,000s)", "MedInc",   fmt="%.2f", step=0.1)
house_age = slider("Median house age (years)",        "HouseAge", fmt="%.1f", step=1.0)
ave_rooms = slider("Avg rooms per household",         "AveRooms", fmt="%.2f", step=0.1)
ave_bedrm = slider("Avg bedrooms per household",      "AveBedrms",fmt="%.2f", step=0.05)
population = slider("Population",                     "Population", fmt="%.0f", step=10.0)
ave_occup = slider("Avg household size",              "AveOccup", fmt="%.2f", step=0.1)
latitude  = slider("Latitude",                        "Latitude", fmt="%.2f", step=0.01)
longitude = slider("Longitude",                       "Longitude",fmt="%.2f", step=0.01)

# Build the input row in the exact column order the model expects
X_input = pd.DataFrame([{
    "MedInc": med_inc,
    "HouseAge": house_age,
    "AveRooms": ave_rooms,
    "AveBedrms": ave_bedrm,
    "Population": population,
    "AveOccup": ave_occup,
    "Latitude": latitude,
    "Longitude": longitude,
}])[FEATURE_NAMES]

# ----------------------------------------------------------------------
# Prediction + uncertainty
# ----------------------------------------------------------------------
prediction = float(model.predict(X_input)[0])

# 95% and 80% residual-based prediction intervals
half95 = 1.96 * RESIDUAL_STD
half80 = 1.28 * RESIDUAL_STD
lo95, hi95 = prediction - half95, prediction + half95
lo80, hi80 = prediction - half80, prediction + half80

near_ceiling = prediction >= 4.5  # within $50k of the $500k top-code

# ----------------------------------------------------------------------
# Main panel
# ----------------------------------------------------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Prediction")
    st.metric(
        "Predicted median home value",
        f"${prediction*100_000:,.0f}",
        help="Random Forest point prediction for a block group with these characteristics.",
    )
    st.markdown(
        f"**95% prediction interval:** &nbsp;&nbsp;${lo95*100_000:,.0f} — ${hi95*100_000:,.0f}  \n"
        f"**80% prediction interval:** &nbsp;&nbsp;${lo80*100_000:,.0f} — ${hi80*100_000:,.0f}"
    )

    if near_ceiling:
        st.warning(
            "⚠️ **Top-code warning.** This prediction is near the $500,000 ceiling "
            "imposed on the 1990 Census target variable. The true uncertainty is "
            "wider than the interval shown — treat as 'high-value, manual review' "
            "rather than a precise estimate."
        )

    st.caption(
        f"Intervals use a residual-based approximation "
        f"(σ = {RESIDUAL_STD:.3f} on $100k scale). Empirical coverage on the held-out "
        "test set is ≈93% for the nominal 95% interval — slightly optimistic because "
        "residuals are heteroscedastic (variance grows in high-value blocks)."
    )

with right:
    st.subheader("Where this prediction sits")
    fig, ax = plt.subplots(figsize=(7, 4))

    # Reference distribution: bell-ish illustration centered on prediction
    grid = np.linspace(max(0, prediction - 4*RESIDUAL_STD),
                       min(5.5, prediction + 4*RESIDUAL_STD), 400)
    pdf = np.exp(-0.5 * ((grid - prediction) / RESIDUAL_STD) ** 2)
    ax.fill_between(grid, pdf, alpha=0.25, color="#2ca02c", label="Approx. predictive density")

    # 95% PI shading
    pi_grid = np.linspace(lo95, hi95, 200)
    pi_pdf = np.exp(-0.5 * ((pi_grid - prediction) / RESIDUAL_STD) ** 2)
    ax.fill_between(pi_grid, pi_pdf, alpha=0.45, color="#2ca02c", label="95% prediction interval")

    ax.axvline(prediction, color="black", linewidth=2, label=f"Point prediction ({prediction:.2f})")
    ax.axvline(5.0, color="red", linestyle="--", alpha=0.7, label="$500k top-code ceiling")

    ax.set_xlabel("Median home value ($100,000s)")
    ax.set_ylabel("Relative density")
    ax.set_title("Predicted value with uncertainty")
    ax.set_xlim(0, 5.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

# ----------------------------------------------------------------------
# Sensitivity analysis: how prediction moves with median income
# ----------------------------------------------------------------------
st.subheader("Sensitivity: how the prediction moves with median income")
st.caption(
    "Holding all other inputs at the values you set in the sidebar, this shows how the "
    "model's prediction changes as median income varies across its observed range. "
    "Sharp non-linearities here (kinks, plateaus) reflect the Random Forest's response "
    "to the spatial and structural interactions in the training data."
)

income_grid = np.linspace(FMIN["MedInc"], FMAX["MedInc"], 80)
sweep = pd.concat([X_input] * len(income_grid), ignore_index=True)
sweep["MedInc"] = income_grid
sweep_pred = model.predict(sweep)

fig2, ax2 = plt.subplots(figsize=(11, 4))
ax2.plot(income_grid, sweep_pred, color="#1f77b4", linewidth=2)
ax2.fill_between(income_grid, sweep_pred - half95, sweep_pred + half95,
                 alpha=0.18, color="#1f77b4", label="95% prediction interval")
ax2.axvline(med_inc, color="red", linestyle="--", alpha=0.7,
            label=f"Your selection: MedInc = {med_inc:.2f}")
ax2.axhline(5.0, color="red", linestyle=":", alpha=0.5)
ax2.set_xlabel("Median income (tens of $1,000s)")
ax2.set_ylabel("Predicted MedHouseVal ($100,000s)")
ax2.set_title("Prediction as a function of median income (other features fixed)")
ax2.legend()
ax2.grid(True, alpha=0.3)
st.pyplot(fig2, clear_figure=True)

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown("---")
with st.expander("ℹ️ About this app & known limitations"):
    st.markdown(
        """
- **Data vintage.** The model is trained on the 1990 U.S. Census. A working deployment
  in 2026 would require retraining on current data — the predictions here are
  illustrative of methodology, not a real-time market signal.
- **Top-code.** The target was top-coded at $500,000 in the source data; predictions near
  or above that ceiling are systematically biased downward and should not be used to
  rank ultra-high-value blocks.
- **Predictive, not causal.** Feature importances and the sensitivity curve above
  describe the model's *learned associations* — they do **not** estimate the
  counterfactual effect of changing a feature while holding the rest of the world fixed.
  Reading them as causal effects would be an interpretation error.
- **Within-block heterogeneity.** The model predicts the *median* for a block group.
  Individual listings within a block can deviate substantially. Use this tool to flag
  blocks worth investigating, not as a final pricing oracle for any single property.
        """
    )
