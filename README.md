# 🎥 Real-Time AI Cartoonifier & Privacy Middleware System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![React](https://img.shields.io/badge/React-Vite-cyan.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green.svg)

---

## 📌 Overview

An intelligent, edge-based computer vision middleware system designed to reduce video streaming bandwidth and inherently preserve privacy.  

By combining **YOLOv8 semantic object detection** with **MOG2 background subtraction**, this pipeline actively identifies critical foreground subjects (humans, vehicles) and applies non-linear bilateral filtering to degrade and "cartoonify" the surrounding background, significantly reducing data entropy.

---

## 🚀 Key Features

- 🎯 **Semantic Subject Preservation:** YOLOv8 ensures important subjects remain clear  
- ⚡ **Real-Time Texture Deletion:** Bilateral filtering smooths unnecessary background  
- 📉 **Bandwidth Optimization:** Reduces video size by up to **65.1%**  
- 🔒 **Privacy by Design:** Background and sensitive info are obscured  
- 🌐 **Full-Stack Dashboard:** React frontend + Python backend  

---

## 📂 Project Structure


📦 Project Root
┣ 📂 backend # Python/OpenCV Processing Engine
┃ ┣ 📜 server.py # Main backend API server
┃ ┣ 📜 batch_stress_test.py # Benchmark script
┃ ┣ 📜 train_custom_cnn.py # CNN training
┃ ┣ 📜 yolov8n.pt # YOLOv8 weights
┃ ┗ 📜 stress_test_report.txt
┣ 📂 frontend # React Dashboard
┃ ┣ 📂 src
┃ ┣ 📜 package.json
┃ ┗ 📜 vite.config.js
┣ 📜 cartoonifier.py # Core pipeline
┗ 📜 README.md


---

## 🛠️ Installation & Setup

### 🔹 Backend Setup (Python)

```bash
cd backend
pip install opencv-python ultralytics flask numpy

⚠️ Note: GPU execution requires NVIDIA CUDA + cuDNN

Start backend server:

python server.py
🔹 Frontend Setup (React + Vite)
cd frontend
npm install

Start frontend:

npm run dev
📊 Benchmarks & Stress Testing

The system was tested across multiple environments:

Environment	Bandwidth Saved	Speed Improvement	Edges Deleted
Daytime Traffic	65.1%	78.3% Faster	1525
Night Pedestrians	48.4%	68.2% Faster	485
Pitch Black	39.3%	65.2% Faster	19
Rain	26.7%	50.2% Faster	636
Moving Camera	3.7%	47.6% Faster	385

⚠️ Note: Rapid lighting changes reduce performance due to MOG2 limitations.

🧠 Core Architecture
🔹 1. Semantic Validation

YOLOv8 detects objects and extracts bounding boxes.

🔹 2. Motion Extraction

MOG2 generates a motion-based foreground mask.

🔹 3. Mask Fusion

Logical operations combine detection + motion masks.

🔹 4. Entropy Reduction

Bilateral filtering is applied only to background.
