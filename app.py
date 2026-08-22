"""
app.py — California housing price predictor
===========================================
Classroom demo. A student types eight numbers describing a California block
group, presses Predict, and gets an estimated median house value from a linear
regression model loaded out of housing_model.pkl (built by train_model.py).
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

MODEL_PATH = Path(__file__).parent / "housing_model.pkl"

st.set_page_config(
    page_title="California housing price predictor",
    page_icon="🏠",
    layout="centered",
)

# --------------------------------------------------------------------------
# Styling. Ink navy for structure, ocher for the one number that matters,
# tabular monospace figures so the price reads like a figure, not a headline.
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
      :root { --ink:#10243b; --ocher:#b7761c; --sage:#5f7f68; --clay:#9c4a3c;
              --rule:#d8dce2; --muted:#5c6673; }
      .block-container { padding-top: 2.4rem; max-width: 900px; }
      h1, h2, h3 { color: var(--ink); letter-spacing: -0.015em; }
      .eyebrow {
        font: 600 0.72rem/1 ui-monospace, "SFMono-Regular", Menlo, monospace;
        letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted);
        margin-bottom: 0.4rem;
      }
      .result {
        border: 1px solid var(--rule); border-left: 4px solid var(--ocher);
        border-radius: 3px; padding: 1.3rem 1.5rem; background: rgba(0,0,0,0.015);
      }
      .price {
        font: 700 3.2rem/1 ui-monospace, "SFMono-Regular", Menlo, monospace;
        font-variant-numeric: tabular-nums; color: var(--ink); margin: 0;
      }
      .price-sub { color: var(--muted); font-size: 0.88rem; margin-top: 0.5rem; }
      .waiting {
        border: 1px dashed var(--rule); border-radius: 3px; padding: 1.3rem 1.5rem;
        color: var(--muted); font-size: 0.92rem;
      }
      .note {
        border-left: 3px solid var(--ink); padding: 0.15rem 0 0.15rem 0.9rem;
        color: var(--muted); font-size: 0.9rem;
      }
      div[data-testid="stMetricValue"] {
        font-family: ui-monospace, Menlo, monospace; font-variant-numeric: tabular-nums;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifact() -> dict:
    """Load housing_model.pkl, training it first if the file is not there yet."""
    if not MODEL_PATH.exists():
        with st.spinner("No housing_model.pkl found — training it once, this takes a minute…"):
            import train_model

            train_model.main()
    return joblib.load(MODEL_PATH)


try:
    art = load_artifact()
except Exception as exc:  # noqa: BLE001
    st.error(
        "The model file could not be loaded or built. Run `python train_model.py` "
        f"and commit `housing_model.pkl` next to `app.py`.\n\nDetails: {exc}"
    )
    st.stop()

FEATURES: list[str] = art["feature_names"]
META: dict = art["feature_meta"]
METRICS: dict = art["metrics"]
SAMPLE: pd.DataFrame = art["sample_data"]
MODEL = art["model"]


def dollars(value_in_100k: float) -> str:
    return f"${value_in_100k * 100_000:,.0f}"


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("California housing price predictor")
st.markdown(
    '<p class="note">Linear regression trained on the California Housing dataset. '
    "Enter values for one block group — roughly 600 to 3,000 people, not a single "
    "house — and the model estimates the median house value there.</p>",
    unsafe_allow_html=True,
)

if art["sklearn_version"] != sklearn.__version__:
    st.warning(
        f"This pickle was written by scikit-learn {art['sklearn_version']} but "
        f"{sklearn.__version__} is installed here. Pickles are not guaranteed to "
        "load across versions — pin the version in requirements.txt.",
        icon="⚠️",
    )

tab_predict, tab_data = st.tabs(["Predict", "Training data"])

# ---------------------------------------------------------------- Predict --
with tab_predict:
    st.markdown('<div class="eyebrow">Enter the block group values</div>',
                unsafe_allow_html=True)

    # A form batches the eight inputs: nothing is predicted until the button is
    # pressed, so students see input and output as two distinct steps.
    with st.form("prediction_form"):
        values: dict[str, float] = {}
        col_a, col_b = st.columns(2)
        for i, col in enumerate(FEATURES):
            m = META[col]
            target = col_a if i % 2 == 0 else col_b
            values[col] = target.number_input(
                m["label"],
                min_value=m["min"],
                max_value=m["max"],
                value=m["default"],
                step=m["step"],
                format="%.2f",
                help=f"{m['help']} Training range: {m['min']:,.2f} to {m['max']:,.2f}.",
            )
        submitted = st.form_submit_button("Predict price", type="primary")

    st.caption(
        "Fields are limited to the 1st–99th percentile of the training data. "
        "A linear model will happily extrapolate past that, and the answer stops "
        "meaning anything when it does."
    )

    st.divider()

    if not submitted:
        st.markdown(
            '<div class="waiting">Enter values above and press '
            "<strong>Predict price</strong> to see the estimate.</div>",
            unsafe_allow_html=True,
        )
    else:
        x_input = pd.DataFrame([values])[FEATURES]
        price = float(MODEL.predict(x_input)[0])
        rmse = METRICS["rmse"]

        st.markdown(
            f"""
            <div class="result">
              <div class="eyebrow">Predicted median house value</div>
              <p class="price">{dollars(max(price, 0.0))}</p>
              <p class="price-sub">
                Typical error on held-out data is about {dollars(rmse)},
                so read this as roughly {dollars(max(price - rmse, 0))} to
                {dollars(price + rmse)}. The training target is capped at
                $500,001, so the model cannot go meaningfully above that.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if price < 0:
            st.warning(
                "The model returned a negative price. Linear regression is not "
                "bounded below — a production model would clip the output or "
                "regress on log(price) instead."
            )

        # --- how the prediction was assembled -----------------------------
        st.markdown("")
        st.markdown('<div class="eyebrow">How the model got there</div>',
                    unsafe_allow_html=True)

        scaler = MODEL.named_steps["scaler"]
        linreg = MODEL.named_steps["model"]
        z = (x_input.to_numpy()[0] - scaler.mean_) / scaler.scale_
        contrib = linreg.coef_ * z  # exact additive decomposition, in $100,000s

        contrib_df = pd.DataFrame(
            {"Feature": [META[c]["label"] for c in FEATURES], "Effect": contrib * 100_000}
        ).sort_values("Effect")

        st.altair_chart(
            alt.Chart(contrib_df)
            .mark_bar(height=16, cornerRadius=1)
            .encode(
                x=alt.X("Effect:Q", title="Dollars added to or subtracted from the baseline"),
                y=alt.Y("Feature:N", sort=None, title=None),
                color=alt.condition(alt.datum.Effect > 0, alt.value("#5f7f68"), alt.value("#9c4a3c")),
                tooltip=[alt.Tooltip("Feature:N"), alt.Tooltip("Effect:Q", format="$,.0f")],
            )
            .properties(height=250),
            width="stretch",
        )
        st.caption(
            f"The baseline — the intercept, an average block group — is "
            f"{dollars(linreg.intercept_)}. Adding these bars to the baseline "
            "reproduces the prediction exactly. That additivity is the whole "
            "appeal of a linear model."
        )

    st.divider()
    st.markdown('<div class="eyebrow">Model performance on held-out data</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{METRICS['r2']:.3f}")
    c2.metric("RMSE", f"${METRICS['rmse'] * 100_000:,.0f}")
    c3.metric("MAE", f"${METRICS['mae'] * 100_000:,.0f}")
    st.caption(
        f"Fit on {METRICS['n_train']:,} block groups, scored on {METRICS['n_test']:,} "
        "the model never saw. Training scores would be optimistic and are not shown."
    )

    with st.expander("Standardized coefficients"):
        coef = art["coefficients"].copy()
        coef["Feature"] = [META[c]["label"] for c in coef["Feature"]]
        coef["Effect per 1 SD"] = coef["Coefficient"] * 100_000
        st.dataframe(
            coef.style.format({"Coefficient": "{:+.3f}", "Effect per 1 SD": "${:+,.0f}"}),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Inputs were standardized before fitting, so these are directly "
            "comparable: each is the dollar effect of a one standard deviation "
            "increase in that feature, holding the others fixed."
        )

# ---------------------------------------------------------- Training data --
with tab_data:
    st.markdown('<div class="eyebrow">What the model learned from</div>',
                unsafe_allow_html=True)
    st.write(
        f"The training split held **{METRICS['n_train']:,} block groups**; "
        f"**{METRICS['n_test']:,}** more were held out for testing. Below is a "
        f"random sample of {len(SAMPLE)} real training rows."
    )

    n_rows = st.number_input(
        "Rows to show", min_value=5, max_value=len(SAMPLE), value=25, step=5
    )
    show = SAMPLE.head(int(n_rows)).copy()
    show["Price ($)"] = show["MedHouseVal"] * 100_000
    st.dataframe(
        show.drop(columns=["MedHouseVal"]),
        width="stretch",
        hide_index=True,
        column_config={"Price ($)": st.column_config.NumberColumn(format="$%d")},
    )
    st.download_button(
        "Download this sample as CSV",
        SAMPLE.to_csv(index=False).encode(),
        file_name="california_housing_sample.csv",
        mime="text/csv",
    )

    st.divider()
    plot_df = SAMPLE.assign(Price=SAMPLE["MedHouseVal"] * 100_000)

    st.markdown('<div class="eyebrow">Median income vs. house value</div>',
                unsafe_allow_html=True)
    st.altair_chart(
        alt.Chart(plot_df)
        .mark_circle(size=45, opacity=0.55, color="#10243b")
        .encode(
            x=alt.X("MedInc:Q", title="Median income ($10k units)"),
            y=alt.Y("Price:Q", title="Median house value", axis=alt.Axis(format="$,.0f")),
            tooltip=[alt.Tooltip("MedInc:Q", format=".2f"),
                     alt.Tooltip("Price:Q", format="$,.0f")],
        )
        .properties(height=300),
        width="stretch",
    )
    st.caption(
        "Income carries most of the signal, and the flat line of points at "
        "$500,001 is the censored target — a modelling problem, not a data error."
    )

    st.markdown('<div class="eyebrow">Distribution of house values</div>',
                unsafe_allow_html=True)
    st.altair_chart(
        alt.Chart(plot_df)
        .mark_bar(color="#10243b", opacity=0.85)
        .encode(
            x=alt.X("Price:Q", bin=alt.Bin(maxbins=30), title="Median house value",
                    axis=alt.Axis(format="$,.0f")),
            y=alt.Y("count():Q", title="Block groups"),
        )
        .properties(height=280),
        width="stretch",
    )

    st.divider()
    st.markdown('<div class="eyebrow">Summary statistics of the predictors</div>',
                unsafe_allow_html=True)
    st.dataframe(art["summary_stats"].style.format("{:.2f}"), width="stretch")
    st.caption(
        "Rows are block groups of roughly 600–3,000 people, not individual "
        "houses, and the data comes from the 1990 U.S. Census."
    )
    st.divider()

st.markdown(
    """
    <div style="text-align: center; color: #5c6673; font-size: 0.85rem; padding: 1rem 0 0.5rem 0;">
        Developed by <strong>Dr. Jishan Ahmed</strong><br>
        Assistant Professor of Data Science<br>
        Weber State University
    </div>
    """,
    unsafe_allow_html=True,
)