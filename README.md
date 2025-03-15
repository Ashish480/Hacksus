# Hacksus
AI-Powered Autonomous Waste Segregation & Recycling Optimization

Introduction

In India, improper waste segregation and inefficient recycling systems contribute significantly to environmental pollution, landfill overflow, and resource wastage. Traditional waste management methods often fail due to a lack of automation, public awareness, and incentives for responsible disposal.

Our AI-powered solution leverages machine learning and image recognition to automate waste classification, ensuring effective segregation at the point of disposal. By integrating smart waste bins and a gamification-based reward system, we encourage responsible waste disposal while optimizing recycling efficiency.

Features

AI-Based Waste Classification: Utilizes YOLO-based object detection and a trained deep learning model to categorize waste into plastic, paper, organic, and hazardous materials.

Smart Waste Bins: Automated bins segregate waste accurately, minimizing human intervention.

Gamification & Rewards: Users earn eco-points for correct disposal, redeemable for discounts, donations, or incentives.

IoT Integration: Real-time data tracking enhances efficiency and monitoring.

Circular Economy Support: Segregated waste is directed to appropriate recycling industries.

File Structure

|
|-- carbon/                 # Contains dataset for training and evaluation
|-- qr_codes/               # Stores QR codes for tracking waste bins
|-- weights/                # Model weights and training checkpoints
|   |-- YOLO_Custom_v8m.pt  # Custom-trained YOLO model
|-- control.py              # Script for controlling hardware integration
|-- firebaseml.py           # Firebase integration for cloud-based processing
|-- my_trained_model.h5     # Pretrained deep learning model for waste classification
|-- waste_detection.log     # Log file for monitoring system activity
|-- wastepredict.py         # Prediction script for waste classification
|-- yolopredict.py          # YOLO-based waste detection script
|-- yolov8n.pt              # YOLOv8 pretrained weights
|-- README.md               # Project documentation

Technologies Used

Machine Learning & AI: YOLO, TensorFlow, OpenCV

IoT & Cloud: Firebase, ESP32 Integration

Backend: Python, FastAPI

Frontend: React Native (for mobile-based user interaction)

How It Works

Image Capture: Smart waste bins capture images of disposed items.

AI Classification: The system classifies the waste using a trained ML model.

Segregation & Sorting: Waste is sorted into designated bins automatically.

User Engagement: Users earn eco-points via a mobile app.

Recycling Integration: Collected waste is sent to respective recycling facilities.

Installation & Setup

Clone the repository:

git clone https://github.com/your-repo/ai-waste-segregation.git
cd ai-waste-segregation

Install dependencies:

pip install -r requirements.txt

Run the prediction script:

python wastepredict.py

Future Enhancements

Integration with blockchain for transparent waste tracking.

Expansion of waste categories using advanced AI models.

Real-time mobile notifications for users on their waste disposal habits.


License

This project is licensed under the MIT License - see the LICENSE file for details.

Join Us in Building a Cleaner Future! 🌱

Let's work together to create a smarter and more sustainable waste management system. 🚀




