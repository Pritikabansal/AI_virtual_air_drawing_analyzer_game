<p align="center">
  <img src="" alt="AI Virtual Air Drawing Analyser Game Banner" width="100%" />
</p>

<h1 align="center">🖐️ AI Virtual Air Drawing Analyser Game 🎮✍️</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/OpenCV-enabled-lightgrey?style=flat-square" />
  <img src="https://img.shields.io/badge/MediaPipe-integrated-success?style=flat-square" />
  <img src="https://img.shields.io/badge/PyTorch-ResNet50-red?style=flat-square" />
  <img src="https://img.shields.io/badge/WebApp-Flask-orange?style=flat-square" />
</p>

---

## 📌 Case Study — AI Virtual Air Drawing Analyser Game

### 🎯 Project Overview

The AI Virtual Air Drawing Analyser Game is a gesture-based web application that allows users to draw in the air using hand movements and receive real-time AI feedback on the accuracy of their sketches. This project blends computer vision, deep learning, and human-computer interaction to create a hands-free, interactive sketching experience.

---

### 👁️ Problem Statement

How can we enable users to draw virtually—without pen, paper, or screen—and intelligently assess what they have drawn?

Traditional drawing tools require physical input. This project reimagines drawing through gesture recognition, empowering users to interact naturally and receive intelligent feedback from an AI model trained on sketch-style datasets.

---

### 🛠️ Solution

- Hand gestures tracked using MediaPipe (index finger detection)
- Pygame canvas renders strokes in real-time using fingertip motion
- A random reference image (e.g., mango, apple, house) prompts the user to replicate it
- Drawing is saved and evaluated using a ResNet50-based CNN
- Cosine similarity calculates accuracy between user sketch and dataset images
- Prediction label and accuracy (%) are displayed on screen

---

### 📈 Impact

- Achieved 85–95% accuracy for well-drawn sketches
- Responsive gesture tracking with smooth canvas rendering
- AI-powered feedback encourages creative learning
- Touchless interaction promotes accessibility and fun

---

### 🔧 Features

- 🖐️ Hand gesture-based drawing  
- 🎯 AI feedback based on reference image  
- 🧠 ResNet50 + Cosine Similarity for prediction  
- 🧪 Real-time accuracy display  
- 🖼️ Save sketch & display prediction  
- 🌐 Flask-based web UI (HTML/CSS)  
- 🎨 Color selection & clear options

---

### 🚀 Tech Stack

Python · MediaPipe · PyTorch · ResNet50 · OpenCV · Pygame · Flask · HTML/CSS · Cosine Similarity

---

### 🧠 Learning Outcomes

- Real-time gesture detection using computer vision
- Integrating deep learning models into a full-stack application
- Creating responsive UIs for AI-powered systems
- Applying similarity metrics for image comparison

---

### 📂 Use Cases

- 👩‍🎓 Educational tool for sketch learning  
- 🧠 AI + HCI demo application  
- 🎮 Creative sketching game  
- 🤝 Inclusive input system for users with limited mobility

---

### 🔮 Future Scope

- 🎤 Voice command integration  
- 🤖 Intelligent feedback with shape improvement tips  
- 📱 Mobile app deployment (React Native/Flutter)  
- 👥 Multiplayer or collaborative drawing mode  
- 🕶️ AR/VR sketching in 3D space

---

### 💡 Why This Project Stands Out

This isn’t just a drawing app—it’s a showcase of how gesture recognition, machine learning, and creative interaction can work together to deliver an innovative user experience. The system is scalable, educational, and adaptable across platforms.

🛠️ Built with passion for real-time interaction, creative technology, and user-focused design.

---

🔗 Feel free to fork, explore, or contribute!  
⭐ Star the repo if you liked it!

