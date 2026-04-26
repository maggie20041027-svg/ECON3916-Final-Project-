# California Housing Predictor — ECON 3916 Final Project

**Predicting median home values across California census block groups** to help a local real estate investor benchmark listings against demographic and housing fundamentals.

- **Live dashboard:** https://YOUR-APP-NAME.streamlit.app *(replace after deploying — see step 6 below)*
- **Course:** ECON 3916 (Spring 2026) — Final Project (25% of grade)
- **Author:** Maggie Ma

---

## What this project does

A Random Forest regressor (200 trees, `random_state=42`) trained on scikit-learn's California Housing dataset (Pace & Barry, 1997; 20,640 census block groups; 1990 U.S. Census). Given eight block-level features — median income, housing stock, density, geography — the model predicts the block's median home value with a residual-based prediction interval.

**This is a prediction project, not a causal analysis.** Feature importances describe the model's learned associations, *not* counterfactual effects of changing any single feature. See the report for full caveats.

## Headline results

| Model | Test RMSE | Test R² | 5-fold CV R² |
|---|---|---|---|
| Linear Regression (baseline) | 0.7456 (~$75k) | 0.5758 | ≈0.61 ± 0.01 |
| **Random Forest (final)** | **≈0.503 (~$50k)** | **≈0.806** | **≈0.80 ± 0.01** |

Random Forest cuts held-out RMSE by ~33% over the linear baseline, primarily by capturing the non-linear coastal price gradient (Latitude × Longitude) that linear regression cannot.

---

## Repository contents

```
.
├── app.py                       # Streamlit dashboard (deployed)
├── final_notebook.ipynb         # Full analysis pipeline
├── artifacts/
│   ├── rf_model.pkl             # Trained Random Forest (saved with joblib)
│   └── model_metadata.pkl       # Residual std + feature ranges for the app
├── requirements.txt             # Pinned dependencies
├── README.md                    # You are here
└── data/                        # No raw files — fetched at runtime via sklearn
```

The dataset is fetched in code through `sklearn.datasets.fetch_california_housing()` — no manual download needed.

---

## Reproducing the results

### 1. Clone and set up

```bash
git clone https://github.com/maggie20041027-svg/ECON3916-Final-Project-.git
cd ECON3916-Final-Project-

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the notebook end-to-end

```bash
jupyter lab final_notebook.ipynb
```

Use **Run All**. All random operations are seeded with `random_state=42` / `np.random.seed(42)` — your numbers should match the README's headline results to four decimal places. The final cells save `artifacts/rf_model.pkl` and `artifacts/model_metadata.pkl`, which the Streamlit app loads.

### 3. Launch the dashboard locally

```bash
streamlit run app.py
```

Opens at <http://localhost:8501>. Adjust the sidebar sliders to describe a block group and watch the prediction + intervals update.

---

## How to deploy your own copy on Streamlit Community Cloud

1. Fork this repo on GitHub.
2. Make sure `artifacts/rf_model.pkl` and `artifacts/model_metadata.pkl` are committed (Git LFS not required — the model is small).
3. Visit <https://streamlit.io/cloud>, sign in with GitHub.
4. Click **New app** → pick the fork → set **Main file path** = `app.py` → **Deploy**.
5. First boot takes ~2 minutes while it builds the environment from `requirements.txt`.
6. The permanent URL pattern is `https://<your-app-name>.streamlit.app` — paste yours into Canvas.

---

## Limitations (read before drawing conclusions)

- **Data vintage.** 1990 Census data. Predictions are not 2026 market values — they illustrate the methodology.
- **Top-coding.** The target is censored at $500,000 (≈5% of rows). Predictions near or above 5.0 are systematically biased downward; the app surfaces a warning when this happens.
- **Heteroscedasticity.** Residuals widen at the top of the price distribution. The 95% prediction interval covers ≈93% of held-out cases — close to nominal but slightly optimistic for high-value blocks.
- **Predictive, not causal.** Don't read coefficients or feature importances as policy levers.

## License

Coursework — no license intended for redistribution.
