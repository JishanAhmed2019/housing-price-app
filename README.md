# California housing price predictor

A classroom demo built on `sklearn.datasets.fetch_california_housing`. A student
types eight numbers describing a California block group, presses **Predict price**,
and a linear regression model estimates the median house value there. A second tab
shows a sample of the training rows the model learned from.

Median house value is a continuous dollar amount, which makes this a regression
problem. Linear regression predicts a number; logistic regression predicts a
probability of class membership and cannot return a price. Worth saying out loud in
class, because the confusion is common.

Built for MATH 4400 at Weber State University.

## Files

```
app.py             Streamlit interface (loads the .pkl, never trains in normal use)
train_model.py     Fits the model, evaluates it, writes housing_model.pkl
requirements.txt   Pinned dependencies
housing_model.pkl  Generated artifact — run train_model.py to create it
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_model.py      # downloads the data, writes housing_model.pkl
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub, `housing_model.pkl` included.
2. At [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and
   choose **Create app**.
3. Point it at this repository, branch `main`, main file `app.py`.
4. It installs from `requirements.txt` and serves the app at
   `https://<app-name>.streamlit.app`.

Every push to `main` redeploys automatically.

Commit the `.pkl`. Training takes about a minute and the first visitor should not
pay for it. If the file is missing, `app.py` trains once at startup as a fallback so
the app still comes up, but every cold restart repeats that work. At roughly 25 KB
the file belongs in git directly — don't let Git LFS pick it up unless you later
swap in a much larger model.

Community Cloud puts apps to sleep after a few days without traffic and wakes them
on the next visit, which takes about thirty seconds. Tell students, so a slow first
load doesn't read as a broken link.

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
`scikit-learn` is pinned exactly in `requirements.txt`. Check your local version
with `python -c "import sklearn; print(sklearn.__version__)"` and make the pin
match before deploying. The app compares the installed version against the one
recorded in the artifact and warns in the interface on mismatch.

## Data caveats worth stating in class

- Rows are **block groups** of roughly 600–3,000 people, not individual houses.
- The target is **censored at $500,001** — the flat ceiling is visible in the
  scatter plot on the Training data tab.
- The data is from the **1990 U.S. Census**. It is a teaching set, not a basis for
  any decision about real property today.
- Input fields are capped at the 1st–99th percentile of the training range. A linear
  model extrapolates without complaint, and the prediction stops meaning anything
  when it does.
- Held-out R² lands near 0.58, so roughly forty percent of the variation is
  unexplained. Useful for discussing what a point prediction does and does not
  promise.

## Extending it

`app.py` reads whatever regressor is in the pickle, so swapping `LinearRegression`
for `Ridge` or `HistGradientBoostingRegressor` in `train_model.py` requires no
interface changes — except the contribution chart on the Predict tab, which assumes
a linear model and would need SHAP for anything else.

## License

MIT.