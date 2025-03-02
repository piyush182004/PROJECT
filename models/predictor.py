import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("cityscape_classifier.h5")

# Load label encoder
label_encoder = joblib.load("label_encoder.pkl")

# Image size
IMG_SIZE = (64, 64)

def predict_image(image_path):
    """Predict the class of an image using the trained model and label encoder."""
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Image not found!", None
    
    img = cv2.resize(img, IMG_SIZE) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Shape: (1, 64, 64, 3)

    # Create a sequence of 5 images for LSTM input
    img_sequence = np.tile(img, (5, 1, 1, 1))  # Shape: (5, 64, 64, 3)
    img_sequence = np.expand_dims(img_sequence, axis=0)  # Shape: (1, 5, 64, 64, 3)

    # Make prediction
    prediction = model.predict(img_sequence)
    predicted_class_index = np.argmax(prediction)
    class_name = label_encoder.inverse_transform([predicted_class_index])[0]
    confidence = np.max(prediction) * 100

    return class_name, confidence

