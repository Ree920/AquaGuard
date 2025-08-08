import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# ======================
# Load models and explainer
# ======================
@st.cache_resource
def load_models():
    pipeline = joblib.load("models/pipeline.joblib")
    shap_data = joblib.load("models/shap_explainer.joblib")
    return pipeline, shap_data

pipeline, shap_data = load_models()
explainer = shap_data["explainer"]
shap_values = shap_data["shap_values"]
feature_names = shap_data["feature_names"]

# ======================
# UI
# ======================
st.set_page_config(page_title="Water Conservation Advisor", layout="centered")
st.title("💧 Water Conservation Recommendation App")
st.write("Enter your household details to get personalized water-saving suggestions.")

with st.form("input_form"):
    household_size = st.number_input("Household size", min_value=1, max_value=10, value=3)
    garden_size = st.selectbox("Garden size", ["none", "small", "medium", "large"])
    has_dishwasher = st.selectbox("Has dishwasher?", ["No", "Yes"])
    washing_frequency = st.number_input("Washing machine usage per week", min_value=1, max_value=20, value=5)
    water_pressure = st.slider("Water pressure (bar)", min_value=1.0, max_value=5.0, step=0.1, value=3.0)
    submitted = st.form_submit_button("Get Recommendations")

if submitted:
    # Prepare input
    input_df = pd.DataFrame([{
        "household_size": household_size,
        "garden_size": garden_size,
        "has_dishwasher": 1 if has_dishwasher == "Yes" else 0,
        "washing_frequency": washing_frequency,
        "water_pressure": water_pressure
    }])

    # Predict
    predictions = pipeline.predict(input_df)[0]

    # Target names (must match training order)
    target_cols = ["shower_shorter", "garden_optimize", "tap_off_brushing", "use_dishwasher", "fix_leaks", "optimize_washer"]

    results = dict(zip(target_cols, predictions))

    st.subheader("✅ Recommended Actions")
    for action, value in results.items():
        if value == 1:
            st.write(f"**{action.replace('_', ' ').capitalize()}**")

    # ======================
    # SHAP explanation
    # ======================
    st.subheader("🔍 Why these recommendations?")
    st.write("Feature importance for the first target model (shower_shorter):")

    # Transform input for SHAP
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(input_df)

    # SHAP values for the first classifier
    shap_vals_first_target = explainer.shap_values(X_transformed)[1]  # Class 1 for first target

    fig, ax = plt.subplots()
    shap.bar_plot(shap_vals_first_target[0], feature_names=feature_names, max_display=10, show=False)
    st.pyplot(fig)
