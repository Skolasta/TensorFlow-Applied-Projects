# Industrial Casting Quality Control

This project focuses on automated quality inspection in industrial manufacturing using Computer Vision and Deep Learning. It detects casting defects from images of industrial components.

## 🚀 Features
- **Image Classification**: A Convolutional Neural Network (CNN) model that predicts whether a cast part is solid (good quality) or defective.
- **Web Application**: An easy-to-use Streamlit dashboard where users can upload an image of a cast part and get near real-time predictions.
- **Threshold System**: Uses sigmoid output probability to classify the industrial parts based on specific confidence scores.

## 🛠️ Tech Stack
- **Deep Learning / Vision**: TensorFlow, Keras
- **Image Processing**: Pillow (PIL), NumPy
- **Web Interface**: Streamlit

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   # Add your git clone command or path here
   ```

2. **Create a virtual environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```
