from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(
    page_title="Food Spoilage | Project Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      :root { --ink: #18253D; --muted: #69758A; --line: #E3E8F0; }
      .stApp { background: #F7F8FC; color: var(--ink); }
      [data-testid="stHeader"] { background: rgba(247,248,252,.96); }
      .block-container { max-width: 1320px; padding-top: 2rem; padding-bottom: 3rem; }
      .masthead {
        background: linear-gradient(112deg, #273C75 0%, #5B4B9A 55%, #B64F7A 100%);
        border-radius: 20px; padding: 2rem 2.1rem; color: white; margin-bottom: 1.4rem;
      }
      .eyebrow { color: #E6E8FF; font-size: .78rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; }
      .masthead h1 { color: white; font-size: 2.35rem; line-height: 1.18; margin: .55rem 0 .55rem; font-weight: 750; }
      .masthead p { color: #F4F2FF; font-size: 1.03rem; margin: 0; max-width: 760px; }
      .section-kicker { color: #6B5CA5; font-size: .78rem; text-transform: uppercase; letter-spacing: .12em; font-weight: 750; margin-bottom: .3rem; }
      .panel { background: white; border: 1px solid var(--line); border-radius: 16px; padding: 1.2rem 1.35rem; }
      .subtle { color: var(--muted); font-size: .94rem; }
      div[data-testid="stMetric"] { background: white; border: 1px solid var(--line); border-radius: 15px; padding: 1.05rem 1.15rem; }
      div[data-testid="stMetricLabel"] { color: #66738A; }
      div[data-testid="stMetricValue"] { color: #273C75; font-weight: 720; }
      div.stButton > button { background: #5B4B9A; color: white; border: none; border-radius: 10px; min-height: 2.9rem; font-weight: 650; }
      div.stButton > button:hover { background: #473879; color: white; border: none; }
      .stTabs [data-baseweb="tab-list"] { gap: .5rem; }
      .stTabs [data-baseweb="tab"] { background: #ECECF8; border-radius: 9px 9px 0 0; padding: .65rem 1.05rem; }
      .stTabs [aria-selected="true"] { background: white !important; }
      h2, h3 { color: #273C75; }
      hr { border-color: var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "food_spoilage.csv"
METRICS_PATH = BASE_DIR / "results" / "baseline_metrics.json"
FEATURES = ["Preservative", "VacuumPress", "ContaminationLevel", "CoolTemp"]
TARGET = "Spoilage"


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_data
def load_metrics(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


@st.cache_resource
def fit_model(data):
    preprocessor = ColumnTransformer([
        ("preservative", OneHotEncoder(handle_unknown="ignore"), ["Preservative"]),
        ("numeric", "passthrough", ["VacuumPress", "ContaminationLevel", "CoolTemp"]),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)),
    ])
    pipeline.fit(data[FEATURES], data[TARGET])
    return pipeline


def find_metric(metrics, model_name, key):
    keys = {"MAE": ["MAE", "mae", "LOOCV MAE"], "RMSE": ["RMSE", "rmse", "LOOCV RMSE"], "R2": ["R2", "r2", "R²", "LOOCV R²"]}
    for candidate in (metrics.get(model_name, {}), metrics.get(model_name.lower(), {}), metrics):
        if isinstance(candidate, dict):
            for name in keys[key]:
                value = candidate.get(name)
                if isinstance(value, (int, float)):
                    return float(value)
    return None


if not DATA_PATH.exists():
    st.error("Could not find data/food_spoilage.csv in the project folder.")
    st.stop()

data = load_data(str(DATA_PATH))
metrics = load_metrics(str(METRICS_PATH))
missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
if missing:
    st.error("Missing required dataset columns: " + ", ".join(missing))
    st.stop()

model = fit_model(data)

st.markdown(
    """
    <div class="masthead">
      <div class="eyebrow">Machine Learning Project · Model & Data</div>
      <h1>Food Spoilage Prediction</h1>
      <p>An experimental study using storage conditions and preservative formulations to estimate recorded spoilage values.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Summary metrics
average_spoilage = data[TARGET].mean()
row_count = len(data)
factor_count = len(FEATURES)
mae = find_metric(metrics, "Random Forest", "MAE")
rmse = find_metric(metrics, "Random Forest", "RMSE")
r2 = find_metric(metrics, "Random Forest", "R2")

summary = st.columns(4)
summary[0].metric("Experimental records", f"{row_count}")
summary[1].metric("Input factors", f"{factor_count}")
summary[2].metric("Mean recorded spoilage", f"{average_spoilage:.2f}")
summary[3].metric("Model", "Random Forest")

st.write("")
tab_overview, tab_prediction, tab_results, tab_data = st.tabs(
    ["Overview", "Prediction", "Model results", "Dataset"]
)

with tab_overview:
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown('<div class="section-kicker">01 / Project overview</div>', unsafe_allow_html=True)
        st.markdown("## What this model studies")
        st.write(
            "The model looks for patterns between four experimental factors and the "
            "recorded spoilage measurement. The factors are preservative formulation, "
            "vacuum pressure, contamination level and cooling temperature."
        )
        st.markdown("### Machine-learning workflow")
        st.markdown(
            """
            **01 — Prepare the data**  
            Use the recorded experiment table as the model input.

            **02 — Train the model**  
            A Random Forest regressor learns relationships between the four factors and spoilage.

            **03 — Evaluate predictions**  
            Leave-one-out cross-validation estimates how closely predictions match held-out observations.

            **04 — Explore a new combination**  
            Select factor settings on the Prediction tab to see the model's estimated value.
            """
        )
    with right:
        st.markdown('<div class="section-kicker">02 / Experimental pattern</div>', unsafe_allow_html=True)
        st.markdown("## Average spoilage by cooling temperature")
        by_temp = data.groupby("CoolTemp", as_index=False)[TARGET].mean().sort_values("CoolTemp")
        fig, ax = plt.subplots(figsize=(7, 4.4))
        colors = ["#4E79A7", "#59A14F", "#E07B54", "#8B78C6"]
        bars = ax.bar(by_temp["CoolTemp"].astype(str), by_temp[TARGET],
                      color=[colors[i % len(colors)] for i in range(len(by_temp))], width=.58)
        ax.set_xlabel("Cooling temperature")
        ax.set_ylabel("Average recorded spoilage")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#D9DFE9")
        ax.grid(axis="y", alpha=.22)
        ax.set_axisbelow(True)
        ax.bar_label(bars, fmt="%.2f", padding=5, color="#27324A", fontsize=10)
        ax.tick_params(axis="both", colors="#69758A")
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Bars show group averages from the experimental records.")

with tab_prediction:
    st.markdown('<div class="section-kicker">03 / Interactive model</div>', unsafe_allow_html=True)
    st.markdown("## Estimate a recorded spoilage value")
    st.write("Choose settings from the experiment and run the model.")
    form_col, info_col = st.columns([1.05, .95], gap="large")
    with form_col:
        with st.container(border=True):
            preservatives = sorted(data["Preservative"].dropna().astype(str).unique())
            vacs = sorted(pd.to_numeric(data["VacuumPress"], errors="coerce").dropna().unique())
            conts = sorted(pd.to_numeric(data["ContaminationLevel"], errors="coerce").dropna().unique())
            temps = sorted(pd.to_numeric(data["CoolTemp"], errors="coerce").dropna().unique())
            preservative = st.selectbox("Preservative formulation", preservatives)
            vacuum = st.select_slider("Vacuum pressure", options=[float(x) for x in vacs], value=float(vacs[0]))
            contamination = st.select_slider("Contamination level", options=[float(x) for x in conts], value=float(conts[0]))
            temperature = st.select_slider("Cooling temperature", options=[float(x) for x in temps], value=float(temps[0]))
            run_prediction = st.button("Run prediction", use_container_width=True)
    with info_col:
        st.markdown(
            """
            <div class="panel">
              <div class="section-kicker">Selected variables</div>
              <h3>Four experimental factors</h3>
              <p class="subtle">The available choices come directly from the source dataset.</p>
              <hr>
              <p><b>Preservative formulation</b><br><span class="subtle">The formulation label used in the experiment.</span></p>
              <p><b>Vacuum pressure</b><br><span class="subtle">One of the pressure settings tested.</span></p>
              <p><b>Contamination level</b><br><span class="subtle">One of the experimental levels recorded.</span></p>
              <p><b>Cooling temperature</b><br><span class="subtle">One of the temperature settings tested.</span></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if run_prediction:
        row = pd.DataFrame([{
            "Preservative": preservative,
            "VacuumPress": vacuum,
            "ContaminationLevel": contamination,
            "CoolTemp": temperature,
        }], columns=FEATURES)
        value = float(model.predict(row)[0])
        st.markdown("### Prediction result")
        result, context = st.columns([.7, 1.3])
        result.metric("Estimated spoilage value", f"{value:.2f}")
        context.info("This value is the model's estimate of the recorded Spoilage measurement for the selected settings.")

with tab_results:
    st.markdown('<div class="section-kicker">04 / Evaluation</div>', unsafe_allow_html=True)
    st.markdown("## Random Forest model results")
    st.write("The scores below are calculated with leave-one-out cross-validation (LOOCV).")
    if mae is not None and rmse is not None and r2 is not None:
        a, b, c = st.columns(3)
        a.metric("MAE", f"{mae:.3f}", help="Mean Absolute Error: average absolute difference between actual and predicted values. Lower is better.")
        b.metric("RMSE", f"{rmse:.3f}", help="Root Mean Squared Error: emphasizes larger prediction errors. Lower is better.")
        c.metric("R²", f"{r2:.3f}", help="R-squared: compares model performance with a simple mean-based reference. Higher is generally better.")
        st.markdown("### How to read the scores")
        st.markdown(
            """
            - **MAE** summarizes the average size of prediction errors.
            - **RMSE** gives larger errors more influence than MAE.
            - **R²** describes how the predictions compare with variation in the recorded values.
            """
        )
    else:
        st.info("Model metric values could not be read from results/baseline_metrics.json.")
    predictions_path = BASE_DIR / "results" / "predictions.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        st.markdown("### Actual and predicted values")
        st.dataframe(predictions, use_container_width=True, hide_index=True)
        st.download_button(
            "Download prediction results",
            data=predictions.to_csv(index=False).encode("utf-8"),
            file_name="model_predictions.csv",
            mime="text/csv",
        )
    actual_plot = BASE_DIR / "results" / "actual_vs_predicted.png"
    if actual_plot.exists():
        st.markdown("### Actual vs. predicted chart")
        st.image(str(actual_plot), use_container_width=True)

with tab_data:
    st.markdown('<div class="section-kicker">05 / Source data</div>', unsafe_allow_html=True)
    st.markdown("## Experimental dataset")
    st.write("Browse the observations used in the project.")
    st.dataframe(data, use_container_width=True, hide_index=True)
    st.download_button(
        "Download dataset (CSV)",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="food_spoilage_data.csv",
        mime="text/csv",
    )
    st.markdown("### Average spoilage by preservative formulation")
    by_pres = data.groupby("Preservative", as_index=False)[TARGET].mean().sort_values("Preservative")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    palette = ["#E07B54", "#8B78C6", "#4E79A7"]
    bars = ax.bar(by_pres["Preservative"].astype(str), by_pres[TARGET],
                  color=[palette[i % len(palette)] for i in range(len(by_pres))], width=.55)
    ax.set_ylabel("Average recorded spoilage")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y", alpha=.22)
    ax.set_axisbelow(True)
    ax.bar_label(bars, fmt="%.2f", padding=4, color="#27324A")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#7A8498;font-size:.85rem;">Food Spoilage Prediction · Machine Learning Project</p>',
    unsafe_allow_html=True,
)
