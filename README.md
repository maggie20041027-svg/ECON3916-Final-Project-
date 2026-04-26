# ECON 3916: ML Prediction Project — Final Project

**Predicting median home values across California census block groups to support a real estate investor's listing-screening decisions.**

- **Course:** ECON 3916 (Spring 2026), Final Project — Capstone
- **Author:** Maggie Ma
- **Live Streamlit dashboard:** https://maggie-final-project.streamlit.app
- **Submitted:** April 26, 2026

---

## 1. Project Summary

This repository contains a complete machine learning prediction pipeline built on the California Housing dataset (Pace and Barry, 1997; 20,640 census block groups; 1990 U.S. Census). The project compares a Linear Regression baseline against a Random Forest regressor and deploys the winning model as an interactive Streamlit dashboard.

The intended user is a local California real estate investor screening listings across the state. Given a block group's eight observable features (median income, housing stock, density, geography), the model returns a predicted median home value with a calibrated 95 percent prediction interval.

This is a **prediction** project, not a causal analysis. Feature importances describe the model's learned associations, not the counterfactual effect of changing any single feature.

## 2. Headline Results

| Model | Test RMSE (95% CI) | Test R² | 5-fold CV R² |
|---|---|---|---|
| Linear Regression (baseline) | 0.7452 [0.710, 0.789] | 0.5758 | 0.6115 ± 0.012 |
| Random Forest (200 trees, full notebook) | 0.5035 [0.482, 0.525] | 0.8062 | 0.8056 ± 0.007 |
| Random Forest (100 trees, deployed in app) | 0.5050 | 0.8054 | — |

The Random Forest cuts held-out RMSE by approximately 33 percent over the linear baseline, primarily by capturing the non-linear coastal price gradient that linear regression cannot represent. Bootstrap 95 percent confidence intervals on the two models' RMSEs do not overlap, confirming the gap is well outside resampling noise.

The deployed dashboard uses a smaller 100-tree variant (compressed to 17 MB) to fit GitHub's 25 MB single-file size limit. Held-out accuracy is essentially identical to the 200-tree version (delta R² < 0.001).

## 3. Repository Contents
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

## 4. Data Access

**No data files are committed to this repository.** The California Housing dataset is fetched at runtime from scikit-learn:

```python
from sklearn.datasets import fetch_california_housing
df = fetch_california_housing(as_frame=True).frame
```

- **Source:** Pace, R. K., and Barry, R. (1997). "Sparse Spatial Autoregressions." *Statistics and Probability Letters*, 33, 291–297.
- **Provenance:** 1990 U.S. Census, aggregated to census block groups
- **Documentation:** https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
- **Access date:** April 19, 2026
- **Size:** 20,640 rows, 8 features, 1 continuous target (`MedHouseVal`, units of $100,000)

This satisfies the assignment's "data folder or download script" requirement: the dataset is auto-downloaded by scikit-learn the first time the notebook is run, with no manual steps for the user.

## 5. Reproducing the Results

### 5.1 Environment setup

```bash
git clone https://github.com/maggie20041027-svg/ECON3916-Final-Project-.git
cd ECON3916-Final-Project-

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 or newer is recommended. The pinned versions in `requirements.txt` are tested with Python 3.11 and 3.14.

### 5.2 Run the notebook end-to-end

```bash
jupyter lab 3916_final_project_starter.ipynb
```

Use **Kernel → Restart and Run All**. The notebook contains all seven parts of the analysis pipeline:

- Part 1: Problem Statement
- Part 2: Data Loading and EDA (six sub-sections covering distributions, missing data, outliers, correlations)
- Part 3: Modeling (train/test split, two models, 5-fold CV, bootstrap CIs)
- Part 4: Feature Importance and Key Visualization
- Part 5: SCR Recommendation
- Part 6: Streamlit Export Guide
- Part 7: AI Methodology Appendix

All random operations are seeded with `random_state=42`. Your numbers should match the Headline Results table to four decimal places. The final cells in Part 4 save `artifacts/rf_model.pkl` and `artifacts/model_metadata.pkl`, which the Streamlit app loads.

### 5.3 Launch the Streamlit dashboard locally

```bash
streamlit run app.py
```

The app opens at http://localhost:8501. Adjust the eight sidebar sliders to describe a block group; the predicted median home value, 95 percent and 80 percent prediction intervals, and an interactive sensitivity chart all update in real time.

## 6. Streamlit Dashboard Features

The deployed dashboard satisfies the three rubric requirements for the Streamlit component:

- **Parameter controls.** Eight sliders in the sidebar, one for each model input feature. Each slider is bounded by the empirical min and max of that feature in the training set.
- **Interactive visualization.** A sensitivity panel shows how the predicted median home value moves as median income varies across its full observed range, with all other inputs held fixed at the user's selected values. The chart updates in real time.
- **Prediction with uncertainty.** Each prediction is displayed as a point estimate (large metric tile) plus an explicit 95 percent and 80 percent residual-based prediction interval. A warning banner triggers automatically when the predicted value is within 0.10 of the $500,000 top-code ceiling, where the empirical interval is known to be optimistic.

## 7. Limitations

Read before drawing conclusions from the model:

- **Data vintage.** The model is trained on 1990 Census data. Predictions are not 2026 market values; they illustrate the methodology only. A production deployment would require retraining on current data (e.g., Zillow ZHVI or recent ACS extracts).
- **Top-coding.** The target is censored at $500,000 (~5 percent of training rows). Predictions near or above 5.0 are systematically biased downward; the dashboard surfaces a warning when this happens.
- **Heteroscedasticity.** Residuals widen at the top of the price distribution. The nominal 95 percent prediction interval covers approximately 93 percent of held-out cases, close to nominal but slightly optimistic for high-value blocks.
- **Predictive, not causal.** Feature importances and coefficients describe model associations, not policy levers. Reading any single feature's contribution as a causal effect would be a textbook omitted-variables error.

## 8. Tools and Versions

- **Python** 3.11+
- **scikit-learn** 1.8.0 (model trained), 1.4.0+ (compatible)
- **Streamlit** 1.31.0+
- **pandas** 2.0.0+, **numpy** 1.24.0+
- **matplotlib** 3.7.0+, **seaborn** 0.12.0+, **joblib** 1.3.0+

See `requirements.txt` for the full pinned list used by Streamlit Cloud deployment.

## 9. Deliverables

The four submission components for ECON 3916 Final Project:

| Deliverable | Location |
|---|---|
| GitHub repository | This repo |
| Streamlit dashboard URL | https://maggie-final-project.streamlit.app |
| 5-page report (PDF) | Submitted on Canvas |
| AI Methodology Appendix (PDF) | Submitted on Canvas |

## License

Coursework, not intended for redistribution.
