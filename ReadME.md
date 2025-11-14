# ✋ Indian Sign Language Recognition (ISL)

A deep-learning–powered **Indian Sign Language (ISL) recognition system** that converts hand gestures into text using computer vision, Mediapipe hand ROI extraction, and a custom-trained CNN.

🚀 **Live Demo:**  
https://indian-sign-language-recognition.streamlit.app

---

## 📌 Overview

This project aims to bridge communication gaps for the hearing-impaired by providing a real-time ISL gesture recognition tool.  
Users can:

- Upload an image  
- Capture a gesture with a webcam  
- See the **preprocessing pipeline** (ROI extraction + edge-based model input)  
- Get the **predicted ISL sign** instantly  

The model supports **36 ISL gesture classes** and achieves **high test accuracy**.

---

## ⭐ Features

### 🔹 Real-time Gesture Recognition
- Upload or capture hand-sign images  
- Fast predictions using a custom CNN  

### 🔹 Advanced Computer Vision Pipeline
- Hand detection + ROI extraction using **Mediapipe**
- Preprocessing using edge detection, resizing, and normalization  

### 🔹 Custom CNN Model
- 36-class architecture  
- Batch normalization + regularization  
- Grad-CAM visualization support  

### 🔹 Streamlit Web App
- Clean, responsive UI  
- Sidebar metrics and performance plots  
- Confusion matrix + example predictions  

---

## 🛠️ Technologies Used

- **Python 3.10**
- **TensorFlow / Keras**
- **OpenCV**
- **Mediapipe**
- **Streamlit**
- **NumPy / Pandas**
- **Matplotlib**
- **Pillow**

---

## 📁 Project Structure

```
sign-language-converter/
├── app.py           # Flask app for live prediction
├── capture.py       # Webcam capture script
├── preprocess.py    # Data preprocessing utilities
├── CNN.ipynb        # Model development & experiments
├── requirements.txt # Python dependencies
├── data/            # Training & validation datasets
├── Models/          # Saved Keras models
├── templates/       # Flask HTML templates
└── README.md        # Project documentation

```

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
    streamlit run streamlit_app/app.py
    ```

### 📊 Model Performance

- **Test Accuracy:** 90.6%
- Performance visualizations available inside ```/files```
- Detailed notebook analysis included


## 🔥 Results

- 36 ISL gestures recognized
- Strong performance across diverse samples
- Robust generalization due to augmentation & regularization
- Grad-CAM explanation included

## ❤️ Acknowledgments

Developed to promote accessibility and support the Deaf/HoH community.

*Built with ❤️ using deep learning and computer vision.*