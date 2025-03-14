import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import warnings

# Suppress warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

# Load your trained model (replace with your model path)
model_path = "/home/ashy/Downloads/my_trained_model.h5"  # Replace with the actual path to your trained model
model = load_model(model_path)

# Class labels corresponding to your training data
class_labels = ['metal', 'glass', 'paper', 'trash', 'cardboard', 'plastic', 'glass']

# Initialize webcam
cap = cv2.VideoCapture(2)

# Ensure webcam is opened
if not cap.isOpened():
    print("❌ Error: Unable to access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to capture image.")
        break

    # Preprocess the frame (resize and rescale)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (224, 224))  # Resize to the input size expected by the model
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Rescale pixel values

    # Make prediction
    predictions = model.predict(img)
    predicted_class = class_labels[np.argmax(predictions)]

    # Display the prediction
    cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the webcam feed with prediction
    cv2.imshow('Waste Classification', frame)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

