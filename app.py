import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

st.title("AI-Based Counter-Drone Detection System")
st.write("Upload a radar micro-Doppler signal CSV file to classify the detected object.")

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(11, 61, 1)),
    layers.MaxPooling2D((2,2), padding='same'),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2), padding='same'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')
])

model.load_weights('model.weights.h5')

classes = ["Drones", "Cars", "People"]
X_min = -190.69
X_max = -38.659

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
