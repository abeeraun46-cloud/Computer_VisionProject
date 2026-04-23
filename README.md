# Crowd Detection & Monitoring System (YOLOv8)

## 📌 Project Overview
This project is a Computer Vision system designed to monitor and manage crowding in retail shops or public spaces. It uses **YOLOv8** to detect people in real-time and triggers an alert based on specific spatial constraints.

## 🚀 Key Features
* **Real-time Detection:** High-speed person detection using YOLOv8.
* **Crowd Alert Logic:** Implemented a "Virtual Line/Threshold" system that triggers a notification when more than two people occupy a specific area.
* **Spatial Monitoring:** Analyzes density within the camera frame to manage occupancy.

## 📂 Project Structure
* `backend/`: Logic for handling video streams and alert triggers.
* `yolov8n.pt`: Pre-trained weights used for person detection.
* `train_model.py`: Script used for fine-tuning or running the inference.

## 🛠️ Tools & Technologies
* **Vision:** YOLOv8 (Ultralytics)
* **Language:** Python
* **Environment:** VS Code & OpenCV
