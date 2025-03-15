import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO
from PIL import Image
import time
import serial
from collections import defaultdict

# Load YOLOv8 models for object detection
model_best = YOLO("/home/ashy/waste-detection/weights/best.pt")  # Your best trained YOLO model
model_last = YOLO("/home/ashy/waste-detection/weights/last.pt")  # Your last trained YOLO model
model_custom = YOLO("/home/ashy/Downloads/YOLO_Custom_v8m.pt")  # Custom YOLO model for better accuracy

# Load classification model for refinement
classification_model = load_model("/home/ashy/Downloads/my_trained_model.h5")

# Define waste-related classes
WASTE_CLASSES = ["bottle", "can", "cup", "plastic bag", "paper", "metal", "glass", "organic", "cardboard", "trash"]
IGNORED_CLASSES = ["battery"]  # Objects to hide from display

# Carbon footprint values (kg CO2e per item)
CARBON_FOOTPRINT = {
    "bottle": 0.08, "can": 0.04,
    "cup": 0.03,
    "plastic bag": 0.06,
    "paper": 0.02,
    "metal": 0.15,
    "glass": 0.2,
    "organic": 0.01,
    "cardboard": 0.05,
    "trash": 0.1,
}

# Serial connection to Arduino
arduino = serial.Serial('/dev/ttyACM0', 9600)  # Adjust the port to your Arduino's serial port

# Image preprocessing function for classification model
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize for model input
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch
    return img_array

# Manually set webcam to video2
cam_index = 2
cap = cv2.VideoCapture(cam_index)

if not cap.isOpened():
    print(f"Error: Cannot open camera at index {cam_index}!")
    exit()

# Global variables for total carbon footprint and highest confidence object
total_carbon_footprint = 0.0
highest_confidence_object = None
highest_confidence = 0.0
start_time = None
highest_confidence_carbon_footprint = 0.0

# Initialize object count dictionary to avoid repeated rotation for the same object
object_count = defaultdict(int)

def control_servo(waste_type):
    """Control the servo motor based on detected waste type."""
    if waste_type == "plastic bag":
        arduino.write(b'90')  # Rotate 90 degrees
        time.sleep(2)
        arduino.write(b'0')  # Return to normal position
    elif waste_type == "paper":
        arduino.write(b'180')  # Rotate 180 degrees
        time.sleep(2)
        arduino.write(b'0')  # Return to normal position
    else:
        arduino.write(b'270')  # Rotate 270 degrees
        time.sleep(2)
        arduino.write(b'0')  # Return to normal position

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Only process every 3rd frame to improve FPS
    if int(time.time() * 1000) % 3 != 0:
        continue

    # Run all YOLO models for detection without verbose output
    results_best = model_best(frame, verbose=False)
    results_last = model_last(frame, verbose=False)
    results_custom = model_custom(frame, verbose=False)

    detected_objects = []

    def process_results(results, color, use_classification=False):
        global total_carbon_footprint, highest_confidence_object, highest_confidence, start_time, highest_confidence_carbon_footprint
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = box.conf[0].item()  # Confidence score
                class_id = int(box.cls[0].item())  # Class ID
                label = model_best.names[class_id]  # Get class name from model

                # Ignore "battery" objects (do not draw bounding box or text)
                if label in IGNORED_CLASSES:
                    continue

                detected_objects.append(label)

                # Update highest confidence object
                if conf > highest_confidence:
                    highest_confidence = conf
                    highest_confidence_object = label
                    highest_confidence_carbon_footprint = CARBON_FOOTPRINT.get(label, 0)  # Get carbon footprint of the object
                    start_time = time.time()  # Start timer for 5 seconds

                # Highlight the highest confidence object with a yellow bounding box
                if label == highest_confidence_object:
                    box_color = (0, 255, 255)  # Yellow for highest confidence object
                else:
                    box_color = color  # Default model color

                # Draw bounding box with label
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # If detected object is a waste item, refine classification
                if use_classification and label in WASTE_CLASSES:
                    cropped_object = frame[y1:y2, x1:x2]  # Crop detected object
                    img_pil = Image.fromarray(cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGB))
                    img_tensor = preprocess_image(img_pil)  # Preprocess for model

                    # Predict refined classification using your model
                    refined_prediction = classification_model.predict(img_tensor)
                    refined_label = WASTE_CLASSES[np.argmax(refined_prediction)]  # Get final label
                    label = refined_label

                # Calculate carbon footprint
                if label in CARBON_FOOTPRINT:
                    total_carbon_footprint += CARBON_FOOTPRINT[label]
                    # Increment the count for the detected object
                    object_count[label] += 1

                    # If object is detected more than 4 times, print it to terminal
                    if object_count[label] > 4:
                        print(f"{label} detected more than 4 times")

                    # Only call the function to control the servo when an object is detected
                    if object_count[label] == 1:  # Only rotate the servo for the first detection
                        control_servo(label)

    # Process detections from all models
    process_results(results_best, (0, 255, 0), use_classification=True)  # Green for best.pt
    process_results(results_last, (255, 0, 0))  # Blue for last.pt
    process_results(results_custom, (0, 0, 255))  # Red for YOLO_Custom_v8m.pt

    # Display detected objects list at the top (excluding "battery")
    detected_text = ", ".join(set(detected_objects)) if detected_objects else "No Objects Detected"
    cv2.putText(frame, detected_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Show the frame
    cv2.imshow("Waste Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

