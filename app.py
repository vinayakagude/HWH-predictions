import streamlit as st
import pandas as pd
import h2o
from sklearn.preprocessing import StandardScaler

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

# Initialize H2O
h2o.init()

# Load model
model_path = "DeepLearning_model_python_1742729893668_59.zip"
try:
    dl_model = h2o.import_mojo(model_path)
    st.success("✅ MOJO model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading the MOJO model: {e}")
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

    st.write("📂 **Loaded Test Case Data:**")  # Debugging
    st.dataframe(df_test_cases)  # Show raw data for debugging

    # Rename columns for consistency
    if "predict" not in df_test_cases.columns or "p1" not in df_test_cases.columns:
        st.error("❌ Missing 'predict' or 'p1' columns in test file.")
        st.stop()

    df_test_cases = df_test_cases.rename(columns={"predict": "Actual", "p1": "Model Probability (grad=1)"})

    # Extract feature columns
    feature_columns = [f"ssf_initial:{col}" if col != "Age_Start" else col for col in input_features]
    
    # Verify feature columns exist
    missing_features = [col for col in feature_columns if col not in df_test_cases.columns]
    if missing_features:
        st.error(f"❌ Missing feature columns in test data: {missing_features}")
        st.stop()

    # Prepare dataset for scaling
    df_features = df_test_cases[feature_columns]

    # Ensure scaler is fitted before use
    scaler = StandardScaler()
    scaler.fit(df_features)  # Fit on test data

    # Transform test cases
    df_scaled = pd.DataFrame(scaler.transform(df_features), columns=feature_columns)

    # Convert to H2OFrame for prediction
    h2o_test_data = h2o.H2OFrame(df_scaled)

    # Run batch predictions
    predictions = dl_model.predict(h2o_test_data).as_data_frame()

    # Debugging step: Show predictions
    st.write("📊 **Predictions Output:**")
    st.dataframe(predictions)

    if "predict" not in predictions.columns:
        st.error("❌ Prediction output does not contain 'predict' column.")
        st.stop()

    # Add predictions to dataframe
    df_test_cases["Predicted"] = predictions["predict"]

    # Ensure 'Actual' and 'Predicted' exist before creating 'Match'
    if "Actual" in df_test_cases.columns and "Predicted" in df_test_cases.columns:
        df_test_cases["Match"] = df_test_cases["Actual"] == df_test_cases["Predicted"]
    else:
        st.error("❌ 'Actual' or 'Predicted' column missing after processing.")
        st.stop()

    # Display table with color highlighting
    st.markdown("## 📊 Model Performance on Test Cases")
    st.dataframe(
        df_test_cases.style.apply(
            lambda x: ["background-color: lightgreen" if v else "background-color: pink" for v in x["Match"]],
            axis=0
        )
    )

except Exception as e:
    st.error(f"❌ Error loading test case file: {e}")
