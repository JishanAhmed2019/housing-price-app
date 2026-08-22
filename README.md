---
title: California Housing Price Predictor
emoji: 🏠
colorFrom: blue
colorTo: yellow
sdk: streamlit
app_file: app.py
pinned: false
license: mit
---

# California housing price predictor

A classroom demo built on `sklearn.datasets.fetch_california_housing`. A student
types eight numbers describing a California block group, presses **Predict price**,
and a linear regression model estimates the median house value there.

Median house value is a continuous dollar amount, which makes this a regression
problem. Linear regression predicts a number; logistic regression predicts a
probability of class membership and cannot return a price. Worth saying out loud
in class, because the confusion is common.

## Files

```
app.py             Streamlit interface (loads the .pkl, never trains in normal use)
train_model.py     Fits the model, evaluates it, writes housing_model.pkl
requirements.txt   Pinned dependencies
housing_model.pkl  Generated artifact — run train_model.py to create it
```

## Run locally

```bash
pip install -r requirements.txt
python train_model.py      # downloads the data, writes housing_model.pkl
streamlit run app.py
```

## Deploy to Hugging Face Spaces

1. Create a Space → SDK **Streamlit**.
2. Push `app.py`, `train_model.py`, `requirements.txt`, `README.md`, **and**
   `housing_model.pkl`.
3. The Space builds and starts on its own.

Commit the `.pkl`. Training takes about a minute and the first visitor should not
pay for it. If the file is missing, `app.py` trains once at startup as a fallback so
the Space still comes up, but every cold restart repeats that work. The file is a
few hundred KB, so Git LFS is unnecessary unless you later swap in a larger model.

## What is inside the pickle

One dictionary, written with `joblib`:

- `model` — the full `Pipeline`, scaler included, so the app never has to remember
  preprocessing steps
- `feature_names`, `feature_meta` — labels, help text, and 1st–99th percentile
  bounds for the input fields
- `metrics` — R², RMSE, MAE on the held-out 20%
- `coefficients` — standardized, so they are comparable across features
- `sample_data` — 500 real training rows for the **Training data** tab
- `summary_stats`, `sklearn_version`, `trained_at`

Pickles are tied to the library version that wrote them, which is why
`scikit-learn` is pinned exactly in `requirements.txt`. The app compares the
installed version against the one recorded in the artifact and warns on mismatch.
Retrain with a different version and you must update the pin.

## Data caveats worth stating in class

- Rows are **block groups** of roughly 600–3,000 people, not individual houses.
- The target is **censored at $500,001** — the flat ceiling is visible in the
  scatter plot on the Training data tab.
- The data is from the **1990 U.S. Census**. It is a teaching set, not a basis for
  any decision about real property today.
- Input fields are capped at the 1st–99th percentile of the training range. A linear
  model extrapolates without complaint, and the prediction stops meaning anything
  when it does.

## Extending it

`app.py` reads whatever regressor is in the pickle, so swapping
`LinearRegression` for `Ridge` or `HistGradientBoostingRegressor` in
`train_model.py` requires no interface changes — except the contribution chart on
the Predict tab, which assumes a linear model and would need SHAP for anything else.