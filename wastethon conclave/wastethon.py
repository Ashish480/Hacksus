import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from PIL import Image
import time
import serial
import firebase_admin
from firebase_admin import credentials, firestore
from pyzbar.pyzbar import decode
from collections import defaultdict
import qrcode  # Added for QR code generation

# Load Firebase credentials
cred = credentials.Certificate("/home/ashy/Downloads/hacksus/serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Machine ID
MACHINE_ID = "1c111ad0-c247-4540-943b-1eba2cdc0314"

# Generate QR code for machine_id
def generate_machine_qr():
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(MACHINE_ID)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    qr_img.save("machine_qr.png")
    print(f"QR code for machine {MACHINE_ID} saved as machine_qr.png")

# Load YOLO models for waste detection
model_best = YOLO("/home/ashy/waste-detection/weights/best.pt")
model_last = YOLO("/home/ashy/waste-detection/weights/last.pt")
model_custom = YOLO("/home/ashy/Downloads/YOLO_Custom_v8m.pt")

# Load classification model
classification_model = load_model("/home/ashy/Downloads/my_trained_model.h5")

# Waste-related classes
WASTE_CLASSES = ["bottle", "can", "cup", "plastic bag", "paper", "metal", "glass", "organic", "cardboard", "trash"]
IGNORED_CLASSES = ["battery"]

# Carbon credit points (kg CO2e avoided per kg of waste * 100 points)
WASTE_POINTS = {
    "plastic": 20,  # ~2 kg CO2e/kg
    "paper": 10,    # ~1 kg CO2e/kg
    "metal": 40,    # ~4 kg CO2e/kg
    "glass": 5,     # ~0.5 kg CO2e/kg
    "organic": 4    # ~0.2 kg CO2e/kg
}

# Waste categories mapping
WASTE_CATEGORY_MAP = {
    "plastic_bottle": "plastic", "can": "metal", "cup": "plastic",
    "plastic bag": "plastic", "paper": "paper", "cardboard": "paper",
    "metal": "metal", "glass": "glass", "organic": "organic", "trash": "organic"
}

# Serial connection to Arduino
try:
    arduino = serial.Serial('/dev/ttyACM0', 9600)
    time.sleep(2)
except Exception as e:
    print(f"Arduino connection failed: {e}")
    arduino = None

# Camera setup
cam_index = 3
cap = cv2.VideoCapture(cam_index)
if not cap.isOpened():
    print(f"Error: Cannot open camera at index {cam_index}!")
    exit()

# Global variables
object_count = defaultdict(int)
user_logged_in = False
current_user_id = None
last_sent_time = 0

def preprocess_image(img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img / 255.0, axis=0)
    return img_array

def detect_waste(image):
    img = cv2.resize(image, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = classification_model.predict(img)
    class_index = np.argmax(prediction[0])
    waste_types = list(WASTE_POINTS.keys())
    return waste_types[class_index] if class_index < len(waste_types) else "unknown"

def record_transaction(user_id, waste_type):
    points = WASTE_POINTS.get(waste_type, 0)
    transaction_data = {
        "userId": user_id if user_id else "anonymous",
        "machineId": MACHINE_ID,
        "wasteType": waste_type,
        "points": points,
        "timestamp": firestore.SERVER_TIMESTAMP,
        "status": "completed"
    }
    db.collection("transactions").document().set(transaction_data)
    print(f"Recorded transaction: {waste_type} for {points} carbon credits")

    if user_id:
        user_ref = db.collection("users").document(user_id)
        user_doc = user_ref.get()
        if user_doc.exists:
            user_ref.update({
                "points": firestore.Increment(points),
                "lastActive": firestore.SERVER_TIMESTAMP
            })
        else:
            user_ref.set({
                "points": points,
                "lastActive": firestore.SERVER_TIMESTAMP
            }, merge=True)

    db.collection("machines").document(MACHINE_ID).update({
        "lastActive": firestore.SERVER_TIMESTAMP,
        "status": "active"
    })
    return points

def process_results(results, frame, color, use_classification=False):
    global last_sent_time, object_count

    detected_waste = None
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            class_id = int(box.cls[0].item())
            label = model_best.names[class_id]

            if label in IGNORED_CLASSES or conf < 0.6:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if use_classification and label in WASTE_CLASSES:
                cropped_obj = frame[y1:y2, x1:x2]
                img_pil = Image.fromarray(cv2.cvtColor(cropped_obj, cv2.COLOR_BGR2RGB))
                img_tensor = preprocess_image(img_pil)
                refined_label = WASTE_CLASSES[np.argmax(classification_model.predict(img_tensor))]
                label = refined_label

            waste_type = WASTE_CATEGORY_MAP.get(label, "unknown")
            if waste_type in WASTE_POINTS and object_count[waste_type] == 0:
                detected_waste = waste_type
                points = record_transaction(current_user_id if user_logged_in else None, waste_type)
                object_count[waste_type] = 10

    current_time = time.time()
    if detected_waste and (current_time - last_sent_time > 5) and arduino:
        arduino.write(f"{detected_waste}\n".encode())
        print(f"Sent to Arduino: {detected_waste}")
        last_sent_time = current_time

    for waste_type in list(object_count.keys()):
        if object_count[waste_type] > 0:
            object_count[waste_type] -= 1

def main():
    global user_logged_in, current_user_id
    print("Waste sorting machine started. Press 'q' to quit.")
    
    # Generate QR code at startup
    generate_machine_qr()

    # Initialize machine status in Firebase
    db.collection("machines").document(MACHINE_ID).set({
        "status": "idle",
        "currentSession": None,
        "lastActive": firestore.SERVER_TIMESTAMP
    }, merge=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            time.sleep(1)
            continue

        # Check machine status for user session
        if not user_logged_in:
            machine_ref = db.collection("machines").document(MACHINE_ID)
            machine_doc = machine_ref.get()
            if machine_doc and machine_doc.exists:  # Fixing this line
                session_user_id = machine_doc.to_dict().get("currentSession")
                if session_user_id:
                    current_user_id = session_user_id
                    user_logged_in = True
                    print(f"User {current_user_id} connected via app scan!")
  
        # Run YOLO models
        results_best = model_best(frame, verbose=False)
        results_last = model_last(frame, verbose=False)
        results_custom = model_custom(frame, verbose=False)

        # Process results
        process_results(results_best, frame, (0, 255, 0), use_classification=True)
        process_results(results_last, frame, (255, 0, 0))
        process_results(results_custom, frame, (0, 0, 255))

        # Display status
        status_text = f"User: {current_user_id}" if user_logged_in else "Scan QR to connect"
        cv2.putText(frame, status_text, (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Waste Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.close()
    print("Machine shutdown complete")

if __name__ == "__main__":
    main()
