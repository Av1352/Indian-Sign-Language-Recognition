# Sign Language Converter

A deep learning application for real-time Indian Sign Language (ISL) recognition, converting hand gestures to text using computer vision and convolutional neural networks.

## 🚩 Overview

This project aims to bridge the communication gap for the hearing-impaired by translating sign gestures from a webcam into text output. The model supports [number of gestures, e.g., 36] ISL classes and achieves high validation accuracy.

## 📊 Key Features

- Real-time hand detection and gesture segmentation using OpenCV & Mediapipe
- CNN trained on a custom dataset covering [number] ISL signs
- Flask web interface for live predictions
- Jupyter Notebook for data preprocessing, augmentation, model training, and evaluation
- Comprehensive error analysis and confusion matrix plots
- Extensible for new signs and dialects

## 🛠 Technologies Used

- Python 3.9+
- TensorFlow 2.x / Keras
- OpenCV
- Mediapipe
- Flask
- Jupyter Notebook
- Docker (optional for containerization)

## 📁 Project Structure

sign-language-converter/
├── app.py             # Flask app for live prediction
├── capture.py         # Webcam capture script
├── preprocess.py      # Data preprocessing utilities
├── CNN.ipynb          # Model development & experiments
├── requirements.txt   # Python dependencies
├── data/              # Training & validation datasets
├── Models/            # Saved Keras models
├── templates/         # Flask HTML templates
└── README.md          # Project documentation


## 🚀 Getting Started

1. **Clone the repo**
    ```
    git clone https://github.com/Av1352/Sign-language-converter.git
    cd Sign-language-converter
    ```

2. **Install requirements**
    ```
    pip install -r requirements.txt
    ```

3. **Launch the application**
   
    ```
    python app.py
    ```

5. **Interact via browser**
    - Access `http://localhost:5000` to use the live sign prediction web interface.

## 🔥 Results

- **Validation accuracy:** 93.4% (36 ISL signs, 3 dialects)
- **Confusion matrix, precision, recall plots provided in `CNN.ipynb`**
