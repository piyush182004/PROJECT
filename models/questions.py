import os
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Load Model & Label Encoder
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../cityscape_classifier.h5")
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), "../label_encoder.pkl")

model = load_model(MODEL_PATH)
model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# Image size for processing
IMG_SIZE = (64, 64)

def predict_image(image_path):
    """Preprocess image and make a classification prediction"""
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Image not found!", None

    img = cv2.resize(img, IMG_SIZE) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Shape: (1, 64, 64, 3)
    
    # Ensuring the model gets the correct input format
    img_sequence = np.tile(img, (5, 1, 1, 1))  # Shape: (5, 64, 64, 3)
    img_sequence = np.expand_dims(img_sequence, axis=0)  # Shape: (1, 5, 64, 64, 3)

    # Predict
    prediction = model.predict(img_sequence)
    predicted_class = np.argmax(prediction)
    class_name = label_encoder.inverse_transform([predicted_class])[0]
    confidence = np.max(prediction) * 100

    return class_name, confidence
