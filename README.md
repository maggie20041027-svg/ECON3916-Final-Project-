# ECON 3916 — ML Prediction Project

**Predicting California median home values from 1990 Census features.**

Course: ECON 3916 (Spring 2026) · Final Project

## Summary

This repo predicts median house values across California census block groups using 8 demographic and housing features. The model is intended as a screening tool for a local real estate investor benchmarking listings against what a block's demographic and housing fundamentals predict; it is a **predictive** model, not a causal one.

**Dataset:** California Housing — scikit-learn's `fetch_california_housing`, derived from the 1990 U.S. Census (Pace & Barry 1997). 20,640 observations, 8 features, continuous target in units of $100,000.

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
