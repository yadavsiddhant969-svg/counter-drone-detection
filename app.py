import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

st.title("AI-Based Counter-Drone Detection System")
st.write("Upload a radar micro-Doppler signal CSV file to classify the detected object.")

model = tf.keras.models.load_model('drone_classifier_model.keras')
classes = ["Drones", "Cars", "People"]

X_min = -130.0
X_max = 0.0

uploaded_file = st.file_uploader("Upload Radar Signal CSV File", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, header=None).values

    if data.shape != (11, 61):
        st.error(f"Error: Expected shape (11, 61), got {data.shape}")
    else:
        data_norm = (data - X_min) / (X_max - X_min)
        data_norm = data_norm.reshape(1, 11, 61, 1)

        prediction = model.predict(data_norm)
        predicted_class = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"Detected: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")

        if predicted_class == "Drones":
            st.warning("⚠️ ALERT: Potential hostile drone detected!")
