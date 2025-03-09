import streamlit as st
import pandas as pd
import h2o

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

# Initialize the H2O cluster (if not already running)
h2o.init()

# ------------------------------------------------------------------
# Load your trained deep learning model that predicts "grad"
# Replace the model_path with the actual path to your saved model.
grad_path = "Grad"  # e.g., "./models/dl_model"
attend_path = "Attend"  # e.g., "./models/dl_model"
try:
    dl_model = h2o.load_model(grad_path)
except Exception as e:
    st.error(f"Error loading the model from {grad_path}: {e}")
    st.stop()

# ------------------------------------------------------------------
st.title("Deep Learning Predictor for 'grad'")
st.write("Enter values for the following input features to predict 'grad':")

# Define the input features (remove the "ssf_initial:" prefix from the original names)
input_features = [
    "adult_education",
    "child_care",
    "community",            # originally "ssf_initial:community"
    "employment",           # originally "ssf_initial:employment"
    "housing",              # originally "ssf_initial:housing"
    "income",               # originally "ssf_initial:income"
    "math_skills",          # originally "ssf_initial:math_skills"
    "mental_health",        # originally "ssf_initial:mental_health"
    "reading_skills",       # originally "ssf_initial:reading_skills"
    "social",               # originally "ssf_initial:social"
    "substance_abuse",      # originally "ssf_initial:substance_abuse"
    "Age_Start"
]

# Create input widgets for each feature.
input_data = {}
for feature in input_features:
    input_data[feature] = st.number_input(f"Enter value for {feature}:", value=0.0)

# When the user clicks the "Predict" button, perform the prediction.
if st.button("Predict"):
    # Convert the user inputs into a pandas DataFrame.
    input_df = pd.DataFrame([input_data])
    st.write("### Input Data")
    st.write(input_df)
    
    # Convert the DataFrame to an H2OFrame.
    h2o_input = h2o.H2OFrame(input_df)
    
    # Make the prediction using your deep learning model.
    prediction = dl_model.predict(h2o_input)
    prediction_df = prediction.as_data_frame()
    
    st.write("### Prediction for 'grad'")
    st.write(prediction_df)
