# Emotion-Based Music Recommendation System 🎵😄😐😢

A deep learning-based system that detects facial emotions in real-time and plays mood-matching music. It combines facial emotion recognition with a personalized music recommendation engine.

## 🔍 Overview

This repository contains two core components:

1. **Facial Emotion Recognition (FER)** – Uses pre-trained deep learning models to classify facial expressions into emotional categories.
2. **Emotion-Based Music Recommendation** – Recommends and plays songs based on the detected emotion using content-based filtering.

---

## 📌 Models Used

### Facial Emotion Recognition:
- Convolutional Neural Network (CNN)
- VGG16
- ResNet50V2
- EfficientNetB0

### Music Recommendation System:
- VGG16 (for emotion classification)
- Content-based filtering (for matching songs to emotions)

---

## 🧠 Platforms Used

- **Facial Emotion Recognition**: [Kaggle](https://www.kaggle.com/) notebooks with GPU support.
- **Emotion-Based Music Recommendation System**: Local Python environment using Jupyter Notebook or VS Code.

---

## 🛠️ Dependencies

Install all requirements from the `requirements.txt` file.

Core libraries used:

- `tensorflow`, `keras` – Deep learning model building
- `numpy`, `pandas` – Data manipulation and analysis
- `matplotlib`, `seaborn` – Data visualization
- `scikit-learn` – Model evaluation tools
- `Pillow`, `OpenCV` – Image processing and webcam support
- `librosa` – Audio feature extraction
- `pygame` – For playing recommended music

---

## 📁 Project Structure

```bash
EmotionMusicRecommender/
├── songs/                  # Folder containing mp3/wav music files
├── emotion_model/          # Pre-trained emotion recognition model
├── requirements.txt        # List of required Python packages
└── music.py                # Main script to run the system

---
