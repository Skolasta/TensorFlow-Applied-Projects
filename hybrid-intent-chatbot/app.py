from pathlib import Path
import numpy as np
import pickle
import re
import tensorflow as tf
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")
import streamlit as st


print("Booting up system...")

base_dir = Path.cwd()
model_path = base_dir / "chatbot_model.keras"
tokenizer_path = base_dir / "tokenizer.pickle"
label_encoder_path = base_dir / "label_encoder.pickle"

# Settings and Cache Configuration
st.set_page_config(page_title="AI Customer Assistant", layout="centered")


# Using st.cache_resource to cache the model and other resources instead of reloading them every time.
# This ensures that they are loaded only once on startup and quickly accessed in subsequent runs.
@st.cache_resource
def load_artifacts():
    print("--- Loading System Artifacts... ---")

    # Load TensorFlow Model
    model = tf.keras.models.load_model(model_path)

    # Load Tokenizer
    with open(tokenizer_path, "rb") as handle:
        tokenizer = pickle.load(handle)

    # Load Label Encoder
    with open(label_encoder_path, "rb") as handle:
        label_encoder = pickle.load(handle)

    # Load LLM (GPT-2) Model (Feel free to upgrade this to a better LLM or use an API)
    generator = pipeline("text-generation", model="distilgpt2")

    print("--- Loading Completed! ---")
    return model, tokenizer, label_encoder, generator


# Loading sequence
try:
    model, tokenizer, label_encoder, generator = load_artifacts()
    st.success("System is Ready! 🚀")  # Display a green success message
except FileNotFoundError:
    st.error(
        "ERROR: Model files not found! Please ensure that .keras and .pickle files are in the same directory."
    )
    st.stop()


# clean_text function removes noise from the text data, helping the model perform better on predictions.
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\{\{.*?\}\}", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def predict_intent(text):
    # Text cleaning
    text = clean_text(text)
    # Tokenization/Sequencing
    seq = tokenizer.texts_to_sequences([text])
    # Padding sequences for uniform input size
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        seq, maxlen=100, padding="post", truncating="post"
    )
    # Prediction
    pred = model.predict(padded)
    class_id = np.argmax(pred, axis=1)
    # Decoding label
    return label_encoder.inverse_transform(class_id)[0]


def generate_llm_response(intent):
    prompt_templates = {
        'cancel_order': 'Customer Support: I understand you want to cancel. To proceed, please provide your ',
        'track_order': 'Customer Support: Let me check the status. Can you share your ',
        'place_order': 'Customer Support: I can help you place a new order. What item are you looking for ',
        'change_order': 'Customer Support: To modify your order, I need to know your ',
        'check_invoice': 'Customer Support: I can help you with your invoice. Please tell me your ',
        'payment_issue': 'Customer Support: I apologize for the payment issue. Could you verify your ',
        'check_payment_methods': 'Customer Support: We accept several payment methods including ',
        'switch_account': 'Customer Support: To switch accounts, verification is needed. Please provide ',
        'recover_password': 'Customer Support: To reset your password, please confirm your ',
        'delete_account': 'Customer Support: I am sorry to see you go. To confirm account deletion, please enter ',
        'create_account': 'Customer Support: To create a new account, I need basic information such as ',
        'edit_account': 'Customer Support: To edit your account details, please provide your ',
        'registration_problems': 'Customer Support: I am sorry you are having registration problems. Can you describe the issue with ',
        'delivery_period': 'Customer Support: Our standard delivery time is usually ',
        'delivery_options': 'Customer Support: We offer several delivery options such as ',
        'change_shipping_address': 'Customer Support: To change your shipping address, please provide your new ',
        'set_up_shipping_address': 'Customer Support: To set up a shipping address, please provide your ',
        'contact_customer_service': 'Customer Support: I will connect you to a human agent immediately. Please wait while ',
        'contact_human_agent': 'Customer Support: I understand you want a human agent. Please hold while I connect you to ',
        'complaint': 'Customer Support: I am very sorry to hear about your experience. Please tell me more about ',
        'review': 'Customer Support: Thank you for your feedback! We would love to hear your thoughts on ',
        'get_refund': 'Customer Support: I can assist with the refund process. Please provide your order number ',
        'check_refund_policy': 'Customer Support: Our refund policy allows returns within ',
        'track_refund': 'Customer Support: To track your refund, please provide your ',
        'check_cancellation_fee': 'Customer Support: To check for cancellation fees, I need your ',
        'newsletter_subscription': 'Customer Support: To manage newsletter subscription, please provide your email address. You can also ',
        'get_invoice': 'Customer Support: To retrieve your invoice, please provide your '
    }

    # Select the appropriate prompt template based on intent
    selected_prompt = prompt_templates.get(intent, prompt_templates.get("fallback", "Customer Support: "))

    # Generate text using GPT-2
    response = generator(
        selected_prompt, max_new_tokens=30, num_return_sequences=1, pad_token_id=50256
    )
    full_text = response[0]["generated_text"]

    # Clean the output and return
    return full_text.replace("Customer Support:", "").strip()


# UI Design

st.title("🤖 AI Support Assistant")
st.markdown("Hello! I am your AI-powered virtual assistant. How can I help you today?")

# Fetch user input
user_input = st.text_input(
    "How can I help you?", placeholder="e.g.: Where is my cargo?"
)

# Submit button logic
if st.button("Send"):
    if user_input:
        # Step 1: Processing animation indicator
        with st.spinner("AI is thinking..."):

            # Step 2: Extract Intent
            intent = predict_intent(user_input)

            # Step 3: Generate the natural language response
            bot_response = generate_llm_response(intent)

        # Show Output Results
        st.info(
            f"Detected Intent: **{intent}**"
        )  # Show the detected intent to the user for debugging/informative purposes

        # Display the bot's response in a chat bubble
        st.chat_message("assistant").write(bot_response)

    else:
        st.warning("Please type a message before sending.")
