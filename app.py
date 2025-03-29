import streamlit as st
import pandas as pd
import h2o
from sklearn.preprocessing import StandardScaler

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

# Initialize H2O cluster
h2o.init()

# Load the trained deep learning model
model_path = "DeepLearning_model_python_1742729893668_59.zip"
try:
    dl_model = h2o.import_mojo(model_path)
    st.success("MOJO model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the MOJO model from {model_path}: {e}")
    st.stop()

# Define input features
input_features = [
    "adult_education", "child_care", "community", "employment", "housing",
    "income", "math_skills", "mental_health", "reading_skills", "social",
    "substance_abuse", "Age_Start"
]

# Load test cases
file_path = "results (4).xlsx"
try:
    xls = pd.ExcelFile(file_path)
    df_test_cases = xls.parse('Sheet1')

    # Extract relevant columns
    feature_columns = [f"ssf_initial:{col}" if col != "Age_Start" else col for col in input_features]
    
    # Standardize column names
    df_test_cases = df_test_cases.rename(columns={"predict": "Actual", "p1": "Model Probability (grad=1)"})

    # Prepare test inputs for scaling
    df_features = df_test_cases[feature_columns]

    # Fit the StandardScaler on the results data
    scaler = StandardScaler()
    scaler.fit(df_features)  # Fit on full dataset

    # Transform test data
    df_scaled = pd.DataFrame(scaler.transform(df_features), columns=feature_columns)

    # Convert to H2OFrame for prediction
    h2o_test_data = h2o.H2OFrame(df_scaled)

    # Run predictions
    predictions = dl_model.predict(h2o_test_data).as_data_frame()
    df_test_cases["Predicted"] = predictions["predict"]

    # Compare actual vs predicted
    df_test_cases["Match"] = df_test_cases["Actual"] == df_test_cases["Predicted"]

    # Display results with highlighting
    st.markdown("## 📊 Model Performance on Test Cases")
    st.dataframe(
        df_test_cases.style.apply(
            lambda x: ["background-color: lightgreen" if v else "background-color: pink" for v in x["Match"]],
            axis=0
        )
    )

except Exception as e:
    st.error(f"Could not load or parse test case file: {e}")

# ---- USER INPUT & PREDICTION ----

# Create user input fields
st.markdown("## 🔢 Provide Input for Prediction")
input_data = {}
cols = st.columns(3)
for i, feature in enumerate(input_features):
    col = cols[i % 3]
    if feature == "Age_Start":
        input_data[feature] = col.number_input(feature, min_value=18, max_value=100, value=25, step=1)
    else:
        input_data[feature] = col.slider(feature, min_value=1, max_value=5, value=3)

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])
input_df.columns = [f"ssf_initial:{col}" if col != "Age_Start" else col for col in input_df.columns]

# Scale user input using the trained transformer
input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_columns)

# Convert to H2OFrame for prediction
h2o_input = h2o.H2OFrame(input_df_scaled)

# Predict when button is clicked
if st.button("Predict"):
    prediction = dl_model.predict(h2o_input).as_data_frame()
    st.markdown("### 🎯 Prediction for 'grad'")
    st.write(prediction)

