import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO
from PIL import Image

# Load YOLOv8 models for object detection
model_best = YOLO("/home/ashy/waste-detection/weights/best.pt")  # Your best trained YOLO model
model_last = YOLO("/home/ashy/waste-detection/weights/last.pt")  # Your last trained YOLO model
model_custom = YOLO("/home/ashy/Downloads/YOLO_Custom_v8m.pt")  # Custom YOLO model for better accuracy

# Load classification model for refinement
classification_model = load_model("/home/ashy/Downloads/my_trained_model.h5")

# Define waste-related classes
WASTE_CLASSES = ["bottle", "can", "cup", "plastic bag", "paper", "metal", "glass", "organic", "cardboard", "trash"]
IGNORED_CLASSES = ["battery"]  # Objects to hide from display

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run all YOLO models for detection
    results_best = model_best(frame)
    results_last = model_last(frame)
    results_custom = model_custom(frame)

    detected_objects = []

    # Function to process results from YOLO models
    def process_results(results, color, use_classification=False):
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

                # Draw bounding box with label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                #cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # If detected object is a waste item, refine classification
                if use_classification and label in WASTE_CLASSES:
                    cropped_object = frame[y1:y2, x1:x2]  # Crop detected object
                    img_pil = Image.fromarray(cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGB))
                    img_tensor = preprocess_image(img_pil)  # Preprocess for model

                    # Predict refined classification using your model
                    refined_prediction = classification_model.predict(img_tensor)
                    refined_label = WASTE_CLASSES[np.argmax(refined_prediction)]  # Get final label

                    # Update bounding box with refined label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow for refined classification
                    cv2.putText(frame, f"{refined_label} ({conf:.2f})", (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Process detections from all models
    process_results(results_best, (0, 255, 0), use_classification=True)  # Green for best.pt
    process_results(results_last, (255, 0, 0))  # Blue for last.pt
    process_results(results_custom, (0, 0, 255))  # Red for YOLO_Custom_v8m.pt

    # Display detected objects list at the top (excluding "battery")
    detected_text = ", ".join(set(detected_objects)) if detected_objects else "No Objects Detected"
    cv2.putText(frame, f"Detected: {detected_text}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("Waste Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

