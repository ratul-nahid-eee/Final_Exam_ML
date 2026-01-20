# app.py
"""
Gradio app to serve the tuned Random Forest pipeline for Water Potability.
Assumes the saved pipeline (preprocessor + RandomForest) is in:
    final_random_forest_model.pkl

Install requirements if you haven't:
    pip install gradio pandas numpy scikit-learn xgboost
"""

import pickle
import numpy as np
import pandas as pd
import gradio as gr

# Path to the saved model (change if necessary)
MODEL_PATH = "final_random_forest_model.pkl"

# Feature names used by the pipeline (order matters)
FEATURE_NAMES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]

def iqr_cap_outliers(X):
    X = np.asarray(X, dtype=float)
    X_capped = X.copy()

    for i in range(X.shape[1]):
        col = X[:, i]

        Q1 = np.percentile(col, 25)
        Q3 = np.percentile(col, 75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        X_capped[:, i] = np.clip(col, lower, upper)

    return X_capped

# Load the pipeline
with open("final_random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)


def make_input_dataframe(
    ph_category,
    hardness,
    solids,
    chloramines,
    sulfate,
    conductivity,
    organic_carbon,
    trihalomethanes,
    turbidity,
):
    """
    Build a single-row DataFrame with the columns expected by the pipeline.
    We will pass 'unknown' ph as np.nan so the pipeline imputer can handle it.
    """
    # Map 'unknown' to actual NaN so SimpleImputer in pipeline imputes it
    if ph_category is None or ph_category == "unknown":
        ph_val = np.nan
    else:
        ph_val = ph_category  # 'acidic', 'neutral', 'alkaline'

    data = {
        "ph": ph_val,
        "Hardness": float(hardness) if hardness is not None else np.nan,
        "Solids": float(solids) if solids is not None else np.nan,
        "Chloramines": float(chloramines) if chloramines is not None else np.nan,
        "Sulfate": float(sulfate) if sulfate is not None else np.nan,
        "Conductivity": float(conductivity) if conductivity is not None else np.nan,
        "Organic_carbon": float(organic_carbon) if organic_carbon is not None else np.nan,
        "Trihalomethanes": float(trihalomethanes) if trihalomethanes is not None else np.nan,
        "Turbidity": float(turbidity) if turbidity is not None else np.nan,
    }

    df = pd.DataFrame([data], columns=FEATURE_NAMES)
    return df


def predict_potability(
    ph_category,
    hardness,
    solids,
    chloramines,
    sulfate,
    conductivity,
    organic_carbon,
    trihalomethanes,
    turbidity,
):
    """
    Predict potability label and probability using the loaded pipeline.
    Returns: (label_str, probability_float)
    """
    # Build input DataFrame
    X = make_input_dataframe(
        ph_category,
        hardness,
        solids,
        chloramines,
        sulfate,
        conductivity,
        organic_carbon,
        trihalomethanes,
        turbidity,
    )

    # Predict probability and label
    try:
        proba = model.predict_proba(X)[:, 1][0]  # probability of class 1 (potable)
        pred = model.predict(X)[0]
    except AttributeError as e:
        # Model doesn't support predict_proba (shouldn't happen for RandomForest pipeline)
        pred = model.predict(X)[0]
        proba = None

    label = "Potable (1)" if int(pred) == 1 else "Not potable (0)"
    proba_display = f"{proba:.3f}" if proba is not None else "N/A"

    # Friendly output string and numeric probability
    label_text = f"{label} — probability: {proba_display}"
    prob_float = float(proba) if proba is not None else None

    return label_text, prob_float


# --- Build Gradio interface ---
title = "Water Potability Predictor"
description = (
    "Enter water measurements. The app uses a pre-trained RandomForest pipeline "
    "to predict whether the water is potable (1) or not (0). "
    "If you don't know a value, leave it blank — the pipeline will impute it."
)

# Input components
inputs = [
    gr.Dropdown(
        choices=["acidic", "neutral", "alkaline", "unknown"],
        value="neutral",
        label="pH category (converted from numeric pH)",
        info="Select acidic / neutral / alkaline or 'unknown' if you don't know the pH.",
    ),
    gr.Number(value=150.0, label="Hardness (mg/L)", precision=3),
    gr.Number(value=10000.0, label="Solids (ppm)", precision=3),
    gr.Number(value=5.0, label="Chloramines (ppm)", precision=3),
    gr.Number(value=300.0, label="Sulfate (mg/L)", precision=3),
    gr.Number(value=400.0, label="Conductivity (µS/cm)", precision=3),
    gr.Number(value=5.0, label="Organic carbon (mg/L)", precision=3),
    gr.Number(value=60.0, label="Trihalomethanes (µg/L)", precision=3),
    gr.Number(value=3.0, label="Turbidity (NTU)", precision=3),
]

# Output components: text summary + numeric probability
outputs = [
    gr.Textbox(label="Prediction"),
    gr.Number(label="Probability (class=1, potable)")
]

# Create interface
iface = gr.Interface(
    fn=predict_potability,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    examples=[
        # (ph_category, hardness, solids, chloramines, sulfate,
        #  conductivity, organic_carbon, trihalomethanes, turbidity)
        ["neutral", 150, 10000, 5, 300, 400, 5, 60, 3],
        ["acidic", 200, 15000, 4.5, 250, 500, 6, 55, 4.0],
        ["unknown", 100, 8000, 3.2, 210, 380, 4.2, 45, 2.5],
    ],
)

iface.launch(share=True)

