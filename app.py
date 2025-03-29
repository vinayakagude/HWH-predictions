import streamlit as st
st.set_page_config(page_title="Grad Predictor", layout="wide")  # ✅ First Streamlit command

import pandas as pd
import h2o
from sklearn.preprocessing import StandardScaler
import os

# -----------------------------
# H2O & Environment Setup
# -----------------------------
# Set Java path for H2O (modify if needed for your system)
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

# Initialize H2O
h2o.init()

# -----------------------------
# Load H2O Model
# -----------------------------
model_path = "DeepLearning_model_python_1742729893668_59.zip"
try:
    dl_model = h2o.import_mojo(model_path)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -----------------------------
# Define Input Features
# -----------------------------
input_features = [
    "adult_education", "child_care", "community", "employment", "housing",
    "income", "math_skills", "mental_health", "reading_skills", "social",
    "substance_abuse", "Age_Start"
]

# -----------------------------
# App UI
# -----------------------------
st.title("🎓 Grad Predictor")
st.markdown("Provide inputs on a scale of **1 to 5** (except Age).")

# Input sliders
input_data = {}
cols = st.columns(3)
for i, feature in enumerate(input_features):
    col = cols[i % 3]
    if feature == "Age_Start":
        input_data[feature] = col.number_input("Age", min_value=18, max_value=100, value=25, step=1)
    else:
        input_data[feature] = col.slider(feature.replace("_", " ").title(), min_value=1, max_value=5, value=3)

st.markdown("---")

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])

        # Rename columns to match model expectation
        input_df.columns = [f"ssf_initial:{col}" if col != "Age_Start" else col for col in input_df.columns]

        # Standardize inputs
        scaler = StandardScaler()
        input_scaled = pd.DataFrame(scaler.fit_transform(input_df), columns=input_df.columns)

        # Convert to H2OFrame
        h2o_input = h2o.H2OFrame(input_scaled)

        # Run prediction
        prediction = dl_model.predict(h2o_input).as_data_frame()

        # Display result
        st.markdown("### 🎯 Prediction Result")
        st.write(prediction)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
