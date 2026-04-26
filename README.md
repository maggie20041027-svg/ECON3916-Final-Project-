# ECON 3916 — ML Prediction Project

**Predicting California median home values from 1990 Census features.**

Course: ECON 3916 (Spring 2026) · Final Project

## Summary

This repo predicts median house values across California census block groups using 8 demographic and housing features. The model is intended as a screening tool for a local real estate investor benchmarking listings against what a block's demographic and housing fundamentals predict; it is a **predictive** model, not a causal one.

**Dataset:** California Housing, scikit-learn's `fetch_california_housing`, derived from the 1990 U.S. Census (Pace & Barry 1997). 20,640 observations, 8 features, continuous target in units of $100,000.

## Repository Contents (checkpoint)

```
.
├── checkpoint.ipynb   # Proposal + EDA + baseline linear model
├── README.md          # This file
└── requirements.txt   # Pinned dependencies
```

The `app.py` (Streamlit dashboard) and the final `report.pdf` will be added for the April 26 submission.

## Quickstart

```bash
# 1. Clone
git clone https://github.com/<your-username>/econ3916-ml-project.git
cd econ3916-ml-project

# 2. Set up environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the notebook
jupyter lab checkpoint.ipynb
```

Or open `checkpoint.ipynb` directly in Google Colab — no local setup required. The dataset is fetched through `sklearn.datasets.fetch_california_housing` on the first run.

## Reproducibility

- All random operations use `random_state=42` / `np.random.seed(42)`.
- The train/test split is 80/20, also with `random_state=42`.
- Baseline linear-regression results on the held-out test set: RMSE ≈ 0.75 ($75k), R² ≈ 0.58. Your re-run should match these to four decimal places.

## Roadmap to April 26

- [ ] Add a second model (Random Forest or Gradient Boosting)
- [ ] 5-fold cross-validation with bootstrapped metric CIs
- [ ] Feature engineering (`rooms_per_person`, log-`Population`)
- [ ] Prediction intervals via quantile regression / conformal prediction
- [ ] Streamlit dashboard (`app.py`) deployed to Community Cloud
- [ ] 5-page SCR report
- [ ] P.R.I.M.E. AI-usage appendix

## License

Coursework — no license.

The dataset is fetched in code through `sklearn.datasets.fetch_california_housing()`. No manual download needed.

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

### 2. Run the notebook

```bash
jupyter lab 3916_final_project_starter.ipynb
```

Use Run All. All random operations are seeded with `random_state=42`. Your numbers should match the headline results above.

The final cells save `artifacts/rf_model.pkl` and `artifacts/model_metadata.pkl`, which the Streamlit app loads.

### 3. Launch the dashboard locally

```bash
streamlit run app.py
```

Opens at http://localhost:8501. Adjust the sidebar sliders to describe a block group and watch the prediction and intervals update.

---

## Limitations

- **Data vintage.** The model is trained on 1990 Census data. Predictions are not 2026 market values; they illustrate the methodology.
- **Top-coding.** The target is censored at $500,000 (~5% of rows). Predictions near or above 5.0 are systematically biased downward; the app surfaces a warning when this happens.
- **Heteroscedasticity.** Residuals widen at the top of the price distribution. The 95% prediction interval covers about 93% of held-out cases, close to nominal but slightly optimistic for high-value blocks.
- **Predictive, not causal.** Feature importances and coefficients describe model associations, not policy levers.

## License

Coursework, not intended for redistribution.
