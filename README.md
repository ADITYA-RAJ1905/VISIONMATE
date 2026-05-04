🚀 VisionMate – AI Assistant for Visually Impaired
🧠 Overview

VisionMate is a real-time AI-powered assistive system designed to help visually impaired individuals navigate safely.
It uses computer vision and deep learning to understand the environment and provide intelligent guidance through visual and audio feedback.

The system combines multiple AI models to detect obstacles, identify walkable paths, estimate depth, and recognize hazards like potholes—all in real time.

✨ Features
🔍 Real-time Object Detection (YOLO)
🛣️ Walkable Path Detection (Semantic Segmentation)
📏 Depth Estimation (MiDaS)
⚠️ Pothole & Hazard Detection (Custom Model)
🧭 Context-Aware Decision System
🎤 Audio Feedback for Navigation Assistance
📷 Live Camera Support (Webcam / Mobile)
🧠 How It Works

The system processes each frame from the camera using multiple AI modules:

Object Detection → Identifies obstacles (people, objects, vehicles)
Segmentation → Determines walkable vs non-walkable regions
Depth Estimation → Calculates distance of obstacles
Pothole Detection → Detects ground-level hazards
Decision Engine → Combines all outputs to generate instructions
🏗️ Tech Stack
Python
PyTorch
OpenCV
Ultralytics YOLO
Hugging Face Transformers (SegFormer)
MiDaS (Depth Estimation)
Streamlit (Web App)
pyttsx3 (Text-to-Speech)
📂 Project Structure
VisionMate/
│
├── app.py                
├── main.ipynb          
├── pothole_model/       
├── datasets/      
├── README.md
└── requirements.txt
🚀 Getting Started
1. Clone the repository
git clone https://github.com/ADITYA-RAJ1905/VISIONMATE
cd VISIONMATE
2. Run the application
streamlit run app.py
🎯 Example Outputs
✅ “Path clear. Move forward”
⚠️ “Pothole ahead. Avoid”
🚫 “Wall ahead. Stop”
⚠️ “Obstacle detected”
📊 Results
Successfully integrates multiple AI models into a unified system
Demonstrates real-time performance on standard hardware
Provides meaningful navigation guidance in dynamic environments
⚠️ Limitations
Performance depends on lighting conditions
Dataset imbalance affects evaluation metrics
Monocular depth estimation is approximate
Occasional false positives in complex scenes
🌍 Impact

VisionMate aims to improve accessibility and independence for visually impaired individuals by providing intelligent, real-time navigation assistance.

🔮 Future Scope
Mobile app / wearable integration
Improved dataset and model accuracy
Multi-sensor fusion (LiDAR, ultrasonic)
Offline optimized deployment
🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

👨‍💻 Author

Aditya Raj

GitHub: https://github.com/ADITYA-RAJ1905
LinkedIn: https://www.linkedin.com/in/aditya-raj-b46481299/
