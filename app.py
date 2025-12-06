

import streamlit as st
import pandas as pd
import joblib
from PIL import Image
from pathlib import Path
import numpy as np

MODEL_PATH = Path("crop_recommendation_model.pkl")
RAINFALL_PATH = Path("district_wise_rainfall_normal.parquet")
LABEL_ENCODER_PATH = Path("label_encoder.pkl")
SCALER_PATH = Path("scaler.pkl")
IMAGE_FILE = Path("assets/crop.jpg")

try:
    model = joblib.load(str(MODEL_PATH))
except Exception as e:
    st.error(f"Failed to load model from {MODEL_PATH}: {e}")
    st.stop()

try:
    district_rainfall_df = pd.read_parquet(str(RAINFALL_PATH))
except Exception as e:
    st.error(f"Failed to load rainfall data from {RAINFALL_PATH}: {e}")
    st.stop()

try:
    label_encoder = joblib.load(str(LABEL_ENCODER_PATH))
except Exception as e:
    st.warning(f"Could not load label encoder from {LABEL_ENCODER_PATH}: {e}")
    label_encoder = None

try:
    scaler = joblib.load(str(SCALER_PATH))
except Exception as e:
    st.warning(f"Could not load scaler from {SCALER_PATH}: {e}")
    scaler = None

def show_homepage():
    if IMAGE_FILE.exists():
        try:
            img = Image.open(IMAGE_FILE)
            st.image(img, use_container_width=True)
        except Exception:
            st.sidebar.warning("Could not load header image.")
    st.title("CROPIQ: AI-Powered Crop Recommendation System for India: District and Month-Specific Insights for Optimized Agricultural Practices")
    st.header("Project Overview")
    st.write("""
        This comprehensive project aims to develop an AI-driven crop recommendation system specifically designed for Indian farmers. 
        By leveraging advanced machine learning algorithms, the system analyzes critical environmental factors such as temperature, 
        humidity, pH, and rainfall to provide district and month-specific crop recommendations. The primary goal is to empower farmers 
        with data-driven insights, thereby enhancing agricultural productivity and sustainability across diverse regions of India. 
        The system ranks the best crops to plant based on the given conditions, ensuring informed decision-making for optimized farming practices.
    """)
    st.header("Project Steps")
    st.write("""
        1. **Data Collection**: Gather data on environmental factors such as temperature, humidity, pH, and rainfall for various districts and months.
        2. **Data Preprocessing**: Clean and preprocess the data to ensure it is suitable for training the machine learning model.
        3. **Model Training**: Train a machine learning model using the preprocessed data to predict the best crops to plant based on the input conditions.
        4. **Model Evaluation**: Evaluate the model's performance using appropriate metrics to ensure its accuracy and reliability.
        5. **Web Interface Development**: Develop a user-friendly web interface using Streamlit to allow farmers to input their data and get crop recommendations.
    """)
    st.header("How to Use")
    st.write("""
        1. Click the "Go to Crop Recommendation" button below.
        2. Select your district and month.
        3. Input the values for Nitrogen (N), Phosphorus (P), Potassium (K), pH, temperature, and humidity.
        4. Click the "Predict" button to get the top crop recommendations based on your input.
    """)
    if st.button("Go to Crop Recommendation"):
        st.session_state.page = "Crop Recommendation"

def get_class_names():
    classes = getattr(model, "classes_", None)
    if classes is None:
        return None
    if label_encoder is not None:
        try:
            return label_encoder.inverse_transform(classes)
        except Exception:
            pass
    return [str(c) for c in classes]

def show_crop_recommendation():
    if IMAGE_FILE.exists():
        try:
            img = Image.open(IMAGE_FILE)
            st.image(img, use_container_width=True)
        except Exception:
            st.sidebar.warning("Could not load header image.")
    st.title("CROPIQ")

    try:
        district = st.selectbox("Select District", district_rainfall_df['DISTRICT'].unique())
    except Exception:
        st.error("Rainfall data does not contain 'DISTRICT' column or is malformed.")
        st.stop()

    month = st.selectbox("Select Month", ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])

    try:
        district_data = district_rainfall_df[district_rainfall_df['DISTRICT'] == district].iloc[0]
        rainfall = float(district_data[month])
    except Exception:
        st.warning("Could not auto-fill rainfall for this district/month. Please enter manually.")
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=10000.0, value=0.0)

    st.write(f"Rainfall (mm): {rainfall}")

    N = st.number_input("Nitrogen (N)", min_value=0, max_value=100, value=50)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=100, value=50)
    K = st.number_input("Potassium (K)", min_value=0, max_value=100, value=50)
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
    temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)

    if st.button("Predict"):
        input_data = pd.DataFrame({
            'N': [N],
            'P': [P],
            'K': [K],
            'temperature': [temperature],
            'humidity': [humidity],
            'ph': [ph],
            'rainfall': [rainfall]
        })

        if scaler is not None:
            try:
                input_data_scaled = scaler.transform(input_data)
            except Exception as e:
                st.error(f"Scaler.transform failed: {e}")
                st.stop()
        else:
            input_data_scaled = input_data.values

        if hasattr(model, "predict_proba"):
            try:
                prediction_probs = model.predict_proba(input_data_scaled)
            except Exception as e:
                st.error(f"model.predict_proba failed: {e}")
                st.stop()
            prediction_classes = getattr(model, "classes_", None)
            class_names = get_class_names()
            if class_names is None:
                st.error("Could not determine class names for predictions.")
                st.stop()
            try:
                prediction_df = pd.DataFrame(prediction_probs, columns=class_names)
            except Exception as e:
                st.error(f"Failed to build prediction DataFrame: {e}")
                st.stop()
            top_n = 5
            top_recommendations = prediction_df.T.sort_values(by=0, ascending=False).head(top_n)
            st.write("Top Crop Recommendations:")
            for i, (crop, score) in enumerate(top_recommendations.iterrows(), start=1):
                st.write(f"{i}. {crop} (Confidence: {score.values[0] * 100:.2f}%)")
        else:
            try:
                preds = model.predict(input_data_scaled)
            except Exception as e:
                st.error(f"model.predict failed: {e}")
                st.stop()
            readable = []
            for p in np.atleast_1d(preds):
                try:
                    if label_encoder is not None:
                        readable.append(label_encoder.inverse_transform([p])[0])
                    else:
                        readable.append(str(p))
                except Exception:
                    readable.append(str(p))
            st.write("Predicted crop(s):")
            for r in readable:
                st.write(f"- {r}")

    if st.button("Back to Home"):
        st.session_state.page = "Home"

if "page" not in st.session_state:
    st.session_state.page = "Home"

if st.session_state.page == "Home":
    show_homepage()
elif st.session_state.page == "Crop Recommendation":
    show_crop_recommendation()
