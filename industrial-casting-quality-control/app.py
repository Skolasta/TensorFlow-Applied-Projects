import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("Quality_model.h5")
    return model


brain = load_model()

st.title("Industrial Casting Quality Control")
loaded_image = st.file_uploader(
    "Upload a photo of a cast part", type=["jpg", "jpeg", "png"]
)
if loaded_image is not None:
    # Open the image and convert it to Grayscale format
    image = Image.open(loaded_image).convert("L")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize the image to match the expected input size of the model (300x300)
    image = image.resize((300, 300))

    # Convert to Numpy array
    image_array = np.array(image)

    # Expand dimensions: (300, 300) -> (1, 300, 300, 1)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = np.expand_dims(image_array, axis=-1)

    # Make prediction (the output of the model will be a probability between 0 and 1)
    prediction = brain.predict(image_array)[0][0]  # Returns a single probability score

    # Business Logic (Sigmoid threshold > 0.5)
    if prediction > 0.5:
        st.success(f"Status: GOOD QUALITY (Thick/Solid) - Confidence Score: {prediction:.4f}")
    else:
        st.error(f"Status: DEFECTIVE QUALITY (Faulty) - Confidence Score: {prediction:.4f}")
