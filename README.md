# 🖐️ OmniTrack: High-Performance Neural Hand Perception

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) 
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**OmniTrack** is a modular, real-time hand tracking and skeletal simulation engine. This project demonstrates advanced coordinate mapping, gesture recognition, and a "Digital Twin" simulation mode, specifically optimized for edge-computing environments.

---

## 🚀 Key Technical Features

### 🔹 Digital Twin Simulation (v3)
OmniTrack renders a synchronized "Mathematical Skeleton" in a dedicated secondary buffer. This isolates the AI's spatial understanding from raw pixel data—a critical step for VR/AR and Robotics integration.

### 🔹 Context-Aware Gesture Engine
Detects complex hand states (Peace signs, Fists, High-Fives) by calculating the Euclidean distance and relative positioning of 21 unique landmarks in real-time.

### 🔹 Mirror-Logic Correction
Features a custom transformation handler for **Handedness Classification**. The system dynamically adjusts logic for Left/Right hand thumb detection to maintain 100% accuracy in mirrored camera views.

### 🔹 Hardware Optimized
* **Inference Speed:** ~30 FPS on dual-core Intel i5-7200U.
* **Architecture:** Utilizes `model_complexity=0` to ensure smooth performance on mobile-grade CPUs without a dedicated GPU.

---

## 📂 System Architecture

```text
.
├──  Mediapipe              # Production-ready hand tracking modules
│   └── hand_tracker.py      # Core detection engine
├──  mediapipe_branch       # Research & Evolution scripts
│   ├── hand_tracker_v1.py   # Basic detection
│   ├── hand_tracker_v2.py   # Individual hand counting & labeling
│   └── hand_tracker_v3.py   # Full Digital Twin simulation
└──  requirements.txt       # Project dependencies
