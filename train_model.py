"""
train_model.py
--------------
Trains a linear regression model on the California Housing dataset
(scikit-learn) and saves it to housing_model.pkl for the Streamlit app.

Median house value is a continuous dollar amount, so this is a regression
problem and linear regression is the right tool: it predicts a number.

Run:  python train_model.py
Out:  housing_model.pkl
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path(__file__).parent / "housing_model.pkl"
RANDOM_STATE = 42

# Human-readable labels and units for the eight predictors. The raw column
# names are terse, and students should never have to guess at units.
FEATURE_LABELS = {
    "MedInc": ("Median income", "Tens of thousands of dollars. 3.5 means $35,000."),
    "HouseAge": ("Median house age (years)", "The dataset caps this at 52 years."),
    "AveRooms": ("Average rooms per household", "Total rooms divided by households."),
    "AveBedrms": ("Average bedrooms per household", "Total bedrooms divided by households."),
    "Population": ("Block group population", "People living in the block group."),
    "AveOccup": ("Average household size", "Population divided by households."),
    "Latitude": ("Latitude", "Degrees north. California spans about 32.5 to 42."),
    "Longitude": ("Longitude", "Degrees west, so negative. About -124 to -114."),
}


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Fetch the California Housing data as a DataFrame plus target Series."""
    bunch = fetch_california_housing(as_frame=True)
    X = bunch.frame.drop(columns=["MedHouseVal"])
    y = bunch.frame["MedHouseVal"]  # median house value in $100,000s
    return X, y


def build_feature_meta(X: pd.DataFrame) -> dict:
    """Input bounds and defaults for the app.

    Bounds come from the 1st and 99th percentile rather than min/max. A few
    block groups report 100+ average rooms per household, and accepting values
    that far outside the training range only produces nonsense predictions.
    """
    meta = {}
    for col in X.columns:
        lo, hi = np.percentile(X[col], [1, 99])
        label, helptext = FEATURE_LABELS[col]
        meta[col] = {
            "label": label,
            "help": helptext,
            "min": float(round(lo, 4)),
            "max": float(round(hi, 4)),
            "default": float(round(X[col].median(), 4)),
            "step": 10.0 if (hi - lo) > 100 else 0.01,
        }
    return meta


def main() -> None:
    print("Downloading California Housing data ...")
    X, y = load_data()
    print(f"  {X.shape[0]:,} block groups, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # StandardScaler is not needed for linear regression to be accurate, but it
    # puts the coefficients on a comparable scale, so each one reads as "effect
    # of a one standard deviation change in that feature."
    model = Pipeline(
        [("scaler", StandardScaler()), ("model", LinearRegression())]
    ).fit(X_train, y_train)

    # --- evaluation on the held-out 20% -----------------------------------
    pred = model.predict(X_test)
    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "r2": float(r2_score(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
    }

    coefficients = pd.DataFrame(
        {"Feature": X.columns, "Coefficient": model.named_steps["model"].coef_}
    )

    # A slice of the real training rows travels with the model so the app can
    # show students what it learned from without re-downloading anything.
    sample = X_train.copy()
    sample["MedHouseVal"] = y_train
    sample = sample.sample(n=min(500, len(sample)), random_state=RANDOM_STATE)

    artifact = {
        "model": model,
        "feature_names": list(X.columns),
        "feature_meta": build_feature_meta(X),
        "metrics": metrics,
        "coefficients": coefficients,
        "sample_data": sample.reset_index(drop=True),
        "summary_stats": X_train.describe().T,
        "sklearn_version": sklearn.__version__,
        "trained_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    joblib.dump(artifact, MODEL_PATH, compress=3)

    print(f"\nSaved -> {MODEL_PATH}  ({MODEL_PATH.stat().st_size / 1024:.0f} KB)")
    print(f"  R2   = {metrics['r2']:.3f}")
    print(f"  RMSE = ${metrics['rmse'] * 100_000:,.0f}")
    print(f"  MAE  = ${metrics['mae'] * 100_000:,.0f}")


if __name__ == "__main__":
    main()