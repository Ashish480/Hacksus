import cv2
import numpy as np
import tensorflow as tf
import firebase_admin
from firebase_admin import credentials, firestore
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import time
import os
from datetime import datetime

# Initialize Firebase
# Replace with your actual path to service account key
SERVICE_ACCOUNT_PATH = "/home/ashy/Downloads/hacksus/serviceAccountKey.json"
if not os.path.exists(SERVICE_ACCOUNT_PATH):
    print(f"Error: Service account key not found at {SERVICE_ACCOUNT_PATH}")
    exit()

try:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase initialized successfully")
except Exception as e:
    print(f"Failed to initialize Firebase: {e}")
    exit()

# Machine ID - Update with your actual machine ID
MACHINE_ID = "1c111ad0-c247-4540-943b-1eba2cdc0314"

# Set machine status to active on startup
try:
    machine_ref = db.collection("machines").document(MACHINE_ID)
    machine_ref.update({
        "status": "active",
        "lastActive": firestore.SERVER_TIMESTAMP,
        "systemStartTime": firestore.SERVER_TIMESTAMP
    })
    print(f"Machine {MACHINE_ID} status set to active")
except Exception as e:
    print(f"Error setting machine status: {e}")

# Load YOLOv8 models for object detection
# Update with your actual model paths
MODEL_PATHS = {
    "best": "/home/ashy/waste-detection/weights/best.pt",
    "last": "/home/ashy/waste-detection/weights/last.pt",
    "custom": "/home/ashy/Downloads/YOLO_Custom_v8m.pt"
}

try:
    model_best = YOLO(MODEL_PATHS["best"])
    model_last = YOLO(MODEL_PATHS["last"])
    model_custom = YOLO(MODEL_PATHS["custom"])
    print("YOLO models loaded successfully")
except Exception as e:
    print(f"Error loading YOLO models: {e}")
    exit()

# Load classification model for refinement
CLASSIFICATION_MODEL_PATH = "/home/ashy/Downloads/my_trained_model.h5"
try:
    classification_model = load_model(CLASSIFICATION_MODEL_PATH)
    print("Classification model loaded successfully")
except Exception as e:
    print(f"Error loading classification model: {e}")
    # Continue without classification model

# Define waste-related classes and carbon footprint values (kg CO2e per item)
WASTE_CLASSES = ["bottle", "can", "cup", "plastic bag", "paper", "metal", "glass", "organic", "cardboard", "trash"]
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

# Configuration options
CONFIG = {
    "confidence_threshold": 0.5,       # Minimum confidence for detection
    "session_timeout_minutes": 2,      # Session expires after this many minutes
    "heartbeat_interval_seconds": 30,  # Update machine status every X seconds
    "camera_id": 0,                    # Camera ID (may need to be changed based on setup)
    "detection_batch_size": 3,         # Process this many detections before updating Firebase
    "debug_mode": True                 # Enable/disable debug messages
}

class WasteDetectionSystem:
    def _init_(self):
        self.last_heartbeat = 0
        self.detection_batch = []
        self.last_session_id = None
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5

    def preprocess_image(self, img):
        """Preprocess image for classification model."""
        try:
            img = img.resize((224, 224))  # Resize for model input
            img_array = image.img_to_array(img) / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch
            return img_array
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None

    def send_heartbeat(self):
        """Update machine status in Firebase periodically."""
        current_time = time.time()
        if current_time - self.last_heartbeat > CONFIG["heartbeat_interval_seconds"]:
            try:
                machine_ref = db.collection("machines").document(MACHINE_ID)
                machine_ref.update({
                    "lastActive": firestore.SERVER_TIMESTAMP,
                    "status": "active",
                    "heartbeat": firestore.SERVER_TIMESTAMP
                })
                if CONFIG["debug_mode"]:
                    print("Heartbeat sent to Firebase")
                self.last_heartbeat = current_time
                self.consecutive_failures = 0
            except Exception as e:
                self.consecutive_failures += 1
                print(f"Error sending heartbeat: {e}")
                if self.consecutive_failures >= self.max_consecutive_failures:
                    print("Too many consecutive failures. Check your Firebase connection.")

    def record_transaction(self, user_id, waste_type):
        """Record transaction in Firebase."""
        try:
            points = CARBON_FOOTPRINT.get(waste_type, 0)  # Use carbon footprint as points
            transaction_data = {
                "userId": user_id,
                "machineId": MACHINE_ID,
                "wasteType": waste_type,
                "points": points,
                "timestamp": firestore.SERVER_TIMESTAMP,
                "status": "completed"
            }
            
            # Add transaction
            transaction_ref = db.collection("transactions").document()
            transaction_ref.set(transaction_data)
            
            # Update user points
            user_ref = db.collection("users").document(user_id)
            user_ref.update({
                "points": firestore.Increment(points),
                "lastActive": firestore.SERVER_TIMESTAMP,
                "totalWasteItems": firestore.Increment(1)
            })
            
            # Update machine status
            machine_ref = db.collection("machines").document(MACHINE_ID)
            machine_ref.update({
                "lastActive": firestore.SERVER_TIMESTAMP,
                "status": "active",
                "lastTransactionTime": firestore.SERVER_TIMESTAMP,
                "processedItems": firestore.Increment(1)
            })
            
            print(f"Transaction recorded: {waste_type}, {points} points")
            return points
        except Exception as e:
            print(f"Error recording transaction: {e}")
            return 0

    def batch_record_transactions(self, user_id):
        """Record multiple transactions at once to reduce Firebase writes."""
        if not self.detection_batch:
            return
            
        try:
            # Count occurrences of each waste type
            waste_counts = {}
            for waste_type in self.detection_batch:
                waste_counts[waste_type] = waste_counts.get(waste_type, 0) + 1
                
            total_points = 0
            
            # Record transactions for each waste type
            for waste_type, count in waste_counts.items():
                points_per_item = CARBON_FOOTPRINT.get(waste_type, 0)
                total_points += points_per_item * count
                
                # Create batch transaction
                transaction_data = {
                    "userId": user_id,
                    "machineId": MACHINE_ID,
                    "wasteType": waste_type,
                    "count": count,
                    "points": points_per_item * count,
                    "timestamp": firestore.SERVER_TIMESTAMP,
                    "status": "completed"
                }
                
                # Add transaction
                transaction_ref = db.collection("transactions").document()
                transaction_ref.set(transaction_data)
            
            # Update user points in a single operation
            user_ref = db.collection("users").document(user_id)
            user_ref.update({
                "points": firestore.Increment(total_points),
                "lastActive": firestore.SERVER_TIMESTAMP,
                "totalWasteItems": firestore.Increment(len(self.detection_batch))
            })
            
            # Update machine status
            machine_ref = db.collection("machines").document(MACHINE_ID)
            machine_ref.update({
                "lastActive": firestore.SERVER_TIMESTAMP,
                "status": "active",
                "lastTransactionTime": firestore.SERVER_TIMESTAMP,
                "processedItems": firestore.Increment(len(self.detection_batch))
            })
            
            print(f"Batch transaction recorded: {len(self.detection_batch)} items, {total_points} total points")
            # Clear the batch after processing
            self.detection_batch = []
            
        except Exception as e:
            print(f"Error recording batch transactions: {e}")

    def detect_waste(self, frame):
        """Detect waste type from image using YOLO model."""
        try:
            results = model_best(frame, verbose=False)
            detected_objects = []
            confidence_scores = {}

            # Process detection results
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    conf = box.conf[0].item()  # Confidence score
                    
                    # Skip if confidence is below threshold
                    if conf < CONFIG["confidence_threshold"]:
                        continue
                        
                    class_id = int(box.cls[0].item())  # Class ID
                    label = model_best.names[class_id]  # Get class name from model

                    # If the object is in our waste classes
                    if label in WASTE_CLASSES:
                        detected_objects.append(label)
                        confidence_scores[label] = conf
                        
                        # Extract the detected object for classification refinement
                        if 'classification_model' in globals():
                            try:
                                # Crop the detected object
                                crop = frame[y1:y2, x1:x2]
                                if crop.size > 0:  # Ensure crop is not empty
                                    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                                    processed_img = self.preprocess_image(crop_pil)
                                    if processed_img is not None:
                                        # Use classification model for refinement
                                        # This would need to be customized based on your classification model
                                        pass
                            except Exception as e:
                                if CONFIG["debug_mode"]:
                                    print(f"Error in classification refinement: {e}")
                        
                        # Draw bounding box for visualization
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return detected_objects, confidence_scores, frame
        except Exception as e:
            print(f"Error in waste detection: {e}")
            return [], {}, frame

    def check_active_session(self):
        """Check if there's an active user session."""
        try:
            machine_ref = db.collection("machines").document(MACHINE_ID)
            machine_data = machine_ref.get().to_dict()
            
            if not machine_data:
                print("Machine not found in Firebase. Make sure the machine ID is correct.")
                return None
            
            current_session = machine_data.get("currentSession", None)
            
            if current_session:
                # Save session ID to compare later
                if current_session != self.last_session_id:
                    print(f"New session detected: {current_session}")
                    self.last_session_id = current_session
                
                # Check if session is expired
                last_active = machine_data.get("lastActive", None)
                if last_active:
                    # Ensure last_active is a datetime object
                    if isinstance(last_active, datetime):
                        # Calculate time difference
                        time_diff = (time.time() - last_active.timestamp()) / 60
                        if time_diff > CONFIG["session_timeout_minutes"]:
                            # Session expired
                            print(f"Session expired after {time_diff:.1f} minutes of inactivity")
                            machine_ref.update({
                                "currentSession": None,
                                "status": "idle",
                                "sessionExpiredAt": firestore.SERVER_TIMESTAMP
                            })
                            self.last_session_id = None
                            return None
                    else:
                        print("Error: lastActive is not a datetime object")
                        return None
            else:
                if self.last_session_id is not None:
                    print("Session ended")
                    self.last_session_id = None
            
            return current_session
        except Exception as e:
            print(f"Error checking active session: {e}")
            return None

    def run(self):
        """Main loop for waste detection and Firebase integration."""
        # Set up webcam for live object detection
        cap = cv2.VideoCapture(CONFIG["camera_id"])

        if not cap.isOpened():
            print(f"Error: Cannot open camera with ID {CONFIG['camera_id']}!")
            return

        print("Waste detection system running. Press 'q' to exit.")

        try:
            while True:
                # Send heartbeat to keep machine status active
                self.send_heartbeat()
                
                # Check for active session
                user_id = self.check_active_session()
                if not user_id:
                    print("Waiting for user to scan QR code...")
                    time.sleep(1)
                    # Still show camera feed while waiting
                    ret, frame = cap.read()
                    if ret:
                        cv2.putText(frame, "Scan QR code to start", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.imshow("Waste Detection", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    continue

                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    time.sleep(0.5)
                    continue

                # Detect waste using YOLOv8
                detected_objects, confidence_scores, annotated_frame = self.detect_waste(frame)

                # If waste is detected, add to batch
                if detected_objects:
                    self.detection_batch.extend(detected_objects)
                    
                    # If batch is full, process it
                    if len(self.detection_batch) >= CONFIG["detection_batch_size"]:
                        self.batch_record_transactions(user_id)

                # Display detected objects on the frame
                detected_text = ", ".join(set(detected_objects)) if detected_objects else "No Objects Detected"
                cv2.putText(annotated_frame, detected_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                # Display session info
                cv2.putText(annotated_frame, f"User: {user_id[:8]}...", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Show the frame
                cv2.imshow("Waste Detection", annotated_frame)

                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            print("Process interrupted by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            # Clean up
            print("Cleaning up resources...")
            cap.release()
            cv2.destroyAllWindows()
            
            # Update machine status to idle on exit
            try:
                machine_ref = db.collection("machines").document(MACHINE_ID)
                machine_ref.update({
                    "status": "idle",
                    "lastActive": firestore.SERVER_TIMESTAMP,
                    "systemShutdownTime": firestore.SERVER_TIMESTAMP
                })
                print("Machine status set to idle")
            except Exception as e:
                print(f"Error updating machine status on exit: {e}")
            
            # Process any remaining detections in batch
            if self.detection_batch and self.last_session_id:
                print(f"Processing remaining {len(self.detection_batch)} detections...")
                self.batch_record_transactions(self.last_session_id)

def listen_for_app_connection():
    """Listen for changes in the machine document to detect app connections."""
    def on_snapshot(doc_snapshot, changes, read_time):
        for doc in doc_snapshot:
            data = doc.to_dict()
            if data and data.get('machineId') == MACHINE_ID:
                current_session = data.get('currentSession')
                status = data.get('status')
                if current_session and status == 'active':
                    print(f"App connected! User session: {current_session}")

    try:
        # Set up listener for machine document
        machine_ref = db.collection("machines").document(MACHINE_ID)
        machine_watch = machine_ref.on_snapshot(on_snapshot)
        print(f"Listening for app connections on machine {MACHINE_ID}")
        return machine_watch
    except Exception as e:
        print(f"Error setting up connection listener: {e}")
        return None

def main():
    """Main entry point for the waste detection system."""
    print("Starting Waste Detection System...")
    
    # Initialize the system
    system = WasteDetectionSystem()
    
    # Start listener for app connections
    connection_watch = listen_for_app_connection()
    
    try:
        # Run the system
        system.run()
    finally:
        # Clean up
        if connection_watch:
            connection_watch.unsubscribe()
        print("System shutdown complete")

if __name__ == "_main_":
    main()
