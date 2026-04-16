# Hybrid Intent Chatbot

This project is an AI-powered customer support chatbot that uses a hybrid approach to intent detection and response generation.

## 🚀 Features
- **Custom Intent Classification**: A Deep Learning (TensorFlow/Keras) model trained to accurately detect user intents from text.
- **Dynamic Response Generation**: Leverages a Large Language Model (`distilgpt2` via Hugging Face `transformers`) to generate natural, contextual replies instead of using static templates.
- **Interactive Web Interface**: A sleek, user-friendly UI built with Streamlit.

## 🛠️ Tech Stack
- **Deep Learning**: TensorFlow, Keras
- **NLP / GenAI**: Hugging Face Transformers
- **Web App**: Streamlit
- **Data Processing**: NumPy, Pickle

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
