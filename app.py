import streamlit as st
import pandas as pd
import h2o

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

# Initialize H2O cluster (if not already running)
h2o.init()

# ------------------------------------------------------------------
# Load your trained deep learning model that predicts "grad"
# Replace with the actual model path from when you saved your model.
model_path = "Grad"
try:
    dl_model = h2o.load_model(model_path)
except Exception as e:
    st.error(f"Error loading the model from {model_path}: {e}")
    st.stop()

# ------------------------------------------------------------------
# Set up page configuration for a cleaner look.
st.set_page_config(page_title="Grad Predictor", layout="wide")
st.title("Grad Predictor")
st.markdown("Provide your inputs on a scale of **1 to 5** to predict the value of **grad**.")

# Define the input feature names (without the "ssf_initial:" prefix)
input_features = [
    "adult_education",
    "child_care",
    "community",         # originally "ssf_initial:community"
    "employment",        # originally "ssf_initial:employment"
    "housing",           # originally "ssf_initial:housing"
    "income",            # originally "ssf_initial:income"
    "math_skills",       # originally "ssf_initial:math_skills"
    "mental_health",     # originally "ssf_initial:mental_health"
    "reading_skills",    # originally "ssf_initial:reading_skills"
    "social",            # originally "ssf_initial:social"
    "substance_abuse",   # originally "ssf_initial:substance_abuse"
    "Age_Start"
]

# Create a dictionary to store the slider inputs.
input_data = {}

# Arrange the 12 sliders into 3 columns for a cleaner layout.
cols = st.columns(3)
for i, feature in enumerate(input_features):
    # Determine which column to use.
    col = cols[i % 3]
    # Use a slider with a range from 1 to 5, defaulting to 3.
    input_data[feature] = col.slider(feature, min_value=1, max_value=5, value=3)

st.markdown("---")

# When the "Predict" button is pressed, perform the prediction.
if st.button("Predict"):
    # Convert the inputs to a pandas DataFrame.
    input_df = pd.DataFrame([input_data])
    st.markdown("### Input Data")
    st.write(input_df)
    
    # Convert the DataFrame to an H2OFrame.
    h2o_input = h2o.H2OFrame(input_df)
    
    # Make the prediction using your loaded deep learning model.
    prediction = dl_model.predict(h2o_input)
    prediction_df = prediction.as_data_frame()
    
    st.markdown("### Prediction for 'grad'")
    st.write(prediction_df)
