import streamlit as st
import pandas as pd
import h2o
from sklearn.preprocessing import StandardScaler
import os

# Ensure Java is set for H2O
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

# Initialize H2O
h2o.init()

# Streamlit app setup
st.set_page_config(page_title="Grad Predictor", layout="wide")
st.title("🎓 Grad Predictor")

# Load trained deep learning model
model_path = "DeepLearning_model_python_1742729893668_59.zip"
try:
    dl_model = h2o.import_mojo(model_path)
    st.success("✅ MOJO model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading MOJO model: {e}")
    st.stop()

# Upload test case file
st.markdown("## 📂 Upload Test Cases (Excel)")
uploaded_file = st.file_uploader("Upload results (4).xlsx", type=["xlsx"])

if uploaded_file:
    try:
        # Load Excel file
        xls = pd.ExcelFile(uploaded_file)
        df_test_cases = xls.parse("Sheet1")  # Change if your sheet name is different

        # Display raw data
        st.write("📊 **Raw Test Case Data:**")
        st.dataframe(df_test_cases.head())

        # Drop unnecessary columns
        df_test_cases = df_test_cases.drop(columns=["Unnamed: 0", "do_not_scale"], errors="ignore")

        # Rename columns for clarity
        df_test_cases = df_test_cases.rename(columns={"predict": "Actual", "p1": "Model Probability (grad=1)"})

        # Extract features
        feature_columns = [col for col in df_test_cases.columns if col not in ["Actual", "Model Probability (grad=1)"]]

        # Ensure all features exist
        missing_features = [col for col in feature_columns if col not in df_test_cases.columns]
        if missing_features:
            st.error(f"❌ Missing feature columns: {missing_features}")
            st.stop()

        # Scale features
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_test_cases[feature_columns]), columns=feature_columns)

        # Convert to H2OFrame
        h2o_test_data = h2o.H2OFrame(df_scaled)

        # Make batch predictions
        predictions = dl_model.predict(h2o_test_data).as_data_frame()

        # Add predictions to dataframe
        df_test_cases["Predicted"] = predictions["predict"]

        # Compare Actual vs. Predicted results
        df_test_cases["Match"] = df_test_cases["Actual"] == df_test_cases["Predicted"]

        # Display processed results with highlights
        st.markdown("## 📊 Model Performance on Test Cases")
        st.dataframe(
            df_test_cases.style.apply(
                lambda x: ["background-color: lightgreen" if v else "background-color: pink" for v in x["Match"]],
                axis=0
            )
        )

        st.success("✅ Predictions completed successfully!")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
