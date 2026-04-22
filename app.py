import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer

MODEL_PATH = "outputs/models/best_model.pkl"
SCALER_PATH = "outputs/models/scaler.pkl"

FEATURE_GROUPS = {
    "Mean values": [
        ("mean radius",             6.98,  28.11, 14.13, "%.3f"),
        ("mean texture",            9.71,  39.28, 19.29, "%.3f"),
        ("mean perimeter",         43.79, 188.50, 91.97, "%.2f"),
        ("mean area",             143.50,2501.00,654.89, "%.1f"),
        ("mean smoothness",         0.053,  0.163,  0.096, "%.4f"),
        ("mean compactness",        0.019,  0.345,  0.104, "%.4f"),
        ("mean concavity",          0.000,  0.427,  0.089, "%.4f"),
        ("mean concave points",     0.000,  0.201,  0.049, "%.4f"),
        ("mean symmetry",           0.106,  0.304,  0.181, "%.4f"),
        ("mean fractal dimension",  0.050,  0.097,  0.063, "%.5f"),
    ],
    "Standard error": [
        ("radius error",            0.112,  2.873,  0.405, "%.4f"),
        ("texture error",           0.360,  4.885,  1.217, "%.4f"),
        ("perimeter error",         0.757, 21.980,  2.866, "%.3f"),
        ("area error",              6.802,542.200, 40.337, "%.2f"),
        ("smoothness error",        0.0017, 0.031,  0.007, "%.5f"),
        ("compactness error",       0.0023, 0.135,  0.025, "%.5f"),
        ("concavity error",         0.000,  0.396,  0.032, "%.5f"),
        ("concave points error",    0.000,  0.053,  0.012, "%.5f"),
        ("symmetry error",          0.0079, 0.079,  0.021, "%.5f"),
        ("fractal dimension error", 0.0009, 0.030,  0.004, "%.6f"),
    ],
    "Worst values": [
        ("worst radius",            7.93,  36.04, 16.27, "%.3f"),
        ("worst texture",          12.02,  49.54, 25.68, "%.3f"),
        ("worst perimeter",        50.41, 251.20,107.26, "%.2f"),
        ("worst area",            185.20,4254.00,880.58, "%.1f"),
        ("worst smoothness",        0.071,  0.223,  0.132, "%.4f"),
        ("worst compactness",       0.027,  1.058,  0.254, "%.4f"),
        ("worst concavity",         0.000,  1.252,  0.272, "%.4f"),
        ("worst concave points",    0.000,  0.291,  0.115, "%.4f"),
        ("worst symmetry",          0.157,  0.664,  0.290, "%.4f"),
        ("worst fractal dimension", 0.055,  0.208,  0.084, "%.5f"),
    ],
}


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def render_sidebar() -> dict:
    st.sidebar.title("Input Features")
    st.sidebar.markdown("Adjust cell nucleus measurements:")
    values = {}
    for group, features in FEATURE_GROUPS.items():
        st.sidebar.markdown(f"**{group}**")
        for name, lo, hi, default, fmt in features:
            values[name] = st.sidebar.slider(
                name, min_value=float(lo), max_value=float(hi),
                value=float(default), format=fmt,
                key=name,
            )
        st.sidebar.markdown("---")
    return values


def predict(model, scaler, values: dict):
    data = load_breast_cancer()
    X = pd.DataFrame([values])[data.feature_names]
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]
    return pred, prob


def main():
    st.set_page_config(
        page_title="Tumor Cell Predictor",
        page_icon="🔬",
        layout="wide",
    )

    st.title("🔬 Breast Cancer Prediction")
    st.markdown(
        "Interactive tool for binary classification of breast cancer cells "
        "(**malignant vs. benign**) using a Logistic Regression model trained on the "
        "[Wisconsin Breast Cancer dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)."
    )

    try:
        model, scaler = load_model()
    except FileNotFoundError:
        st.error(
            "Model not found. Run `python train.py` first to generate "
            "`outputs/models/best_model.pkl` and `outputs/models/scaler.pkl`."
        )
        st.stop()

    values = render_sidebar()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Prediction")
        pred, prob = predict(model, scaler, values)

        if pred == 0:
            st.error("### Malignant")
            st.markdown(f"Confidence: **{prob[0]*100:.1f}%**")
        else:
            st.success("### Benign")
            st.markdown(f"Confidence: **{prob[1]*100:.1f}%**")

        st.markdown("**Class probabilities**")
        prob_df = pd.DataFrame(
            {"Probability": [prob[0], prob[1]]},
            index=["Malignant", "Benign"],
        )
        st.bar_chart(prob_df, color=["#e74c3c"], height=220)

    with col2:
        st.subheader("Input summary")
        summary = pd.DataFrame(
            list(values.items()), columns=["Feature", "Value"]
        )
        # colour rows by group
        group_map = {
            name: group
            for group, features in FEATURE_GROUPS.items()
            for name, *_ in features
        }
        summary["Group"] = summary["Feature"].map(group_map)
        st.dataframe(
            summary.set_index("Feature"),
            use_container_width=True,
            height=500,
        )

    st.markdown("---")
    st.caption(
        "Model: Logistic Regression · ROC-AUC 0.9954 · Accuracy 98.25% · "
        "5-fold stratified CV AUC 0.9957 ± 0.005"
    )


if __name__ == "__main__":
    main()
