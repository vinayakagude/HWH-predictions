import streamlit as st
import pandas as pd
import h2o
from sklearn.preprocessing import StandardScaler


import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

# Initialize H2O cluster (if not already running)
h2o.init()

# ------------------------------------------------------------------
# Load your trained deep learning model that predicts "grad"
# Replace with the actual model path from when you saved your model.

    
    
# ------------------------------------------------------------------
# Set up page configuration for a cleaner look.
st.set_page_config(page_title="Grad Predictor", layout="wide")
st.title("Grad Predictor")
st.markdown("Provide your inputs on a scale of **1 to 5** to predict the value of **grad**.")


model_path = "DeepLearning_model_python_1742729893668_59.zip"
try:
    dl_model = h2o.import_mojo(model_path)
    st.success("MOJO model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the MOJO model from {model_path}: {e}")
    st.stop()
    
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
    col = cols[i % 3]

    if feature == "Age_Start":
        input_data[feature] = col.number_input(feature, min_value=18, max_value=100, value=25, step=1)
    else:
        input_data[feature] = col.slider(feature, min_value=1, max_value=5, value=3)

st.markdown("---")




# When the "Predict" button is pressed, perform the prediction.
if st.button("Predict"):
    # Convert the inputs to a pandas DataFrame.
    input_df = pd.DataFrame([input_data])
    input_df_scaled = input_df.copy()
    input_df_scaled.columns = [
        f"ssf_initial:{col}" if col != "Age_Start" else col for col in input_df.columns
    ]
    scaler = StandardScaler()
    feature_columns = [f"ssf_initial:{col}" if col != "Age_Start" else col for col in input_features]

    # Apply the standard scaler
    input_df_scaled[feature_columns] = scaler.transform(input_df_scaled[feature_columns])

    # Convert to H2OFrame
    h2o_input = h2o.H2OFrame(input_df_scaled)
    
    st.markdown("### Input Data")
    st.write(input_df)
    
    # Convert the DataFrame to an H2OFrame.
    h2o_input = h2o.H2OFrame(input_df_scaled)
    
    # Make the prediction using your loaded deep learning model.
    prediction = dl_model.predict(h2o_input)
    prediction_df = prediction.as_data_frame()
    
    st.markdown("### Prediction for 'grad'")
    st.write(prediction_df)

import pandas as pd

st.markdown("## 🔍 Explore Test Cases")

# Load the test cases Excel file
file_path = "results (4).xlsx"  # If using relative path on deployment
try:
    xls = pd.ExcelFile(file_path)
    df_test_cases = xls.parse('Sheet1')
    
    # Optional: Rename for clarity
    df_display = df_test_cases.rename(columns={
        "predict": "Actual",
        "p1": "Model Probability (grad=1)"
    })

    # Show interactive table
    st.dataframe(df_display[[
        "ssf_initial:adult_education", "ssf_initial:child_care", "ssf_initial:community",
        "ssf_initial:employment", "ssf_initial:housing", "ssf_initial:income",
        "ssf_initial:math_skills", "ssf_initial:mental_health", "ssf_initial:reading_skills",
        "ssf_initial:social", "ssf_initial:substance_abuse", "Age_Start",
        "Actual", "Model Probability (grad=1)"
    ]], use_container_width=True)

    st.caption("Compare model predictions against actual values for each test case.")

except Exception as e:
    st.error(f"Could not load or parse test case file: {e}")
