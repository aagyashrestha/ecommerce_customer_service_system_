import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from gtts import gTTS
import speech_recognition as sr
import io
import torch
import tempfile
import os
from pydub import AudioSegment
from pydub.playback import play
import random

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
nlp = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Define e-commerce specific intents
intents = {
    0: "Product Inquiry",
    1: "Order Status",
    2: "Return Request",
    3: "Payment Issues",
    4: "Shipping Information",
    5: "Account Assistance",
    6: "Cart Management"
}

# Context management to maintain conversation state
conversation_history = {}

# Function for rule-based intent classification
def rule_based_intent(text):
    text = text.lower()
    if any(word in text for word in ["order", "status", "track"]):
        return "Order Status"
    elif any(word in text for word in ["cart", "manage", "items"]):
        return "Cart Management"
    elif any(word in text for word in ["return", "refund", "exchange"]):
        return "Return Request"
    elif any(word in text for word in ["payment", "charge", "billing"]):
        return "Payment Issues"
    elif any(word in text for word in ["shipping", "delivery", "ship"]):
        return "Shipping Information"
    elif any(word in text for word in ["account", "login", "profile"]):
        return "Account Assistance"
    else:
        return None

# Function to classify intent using a hybrid approach
def classify_intent(text):
    intent = rule_based_intent(text)
    if intent is None:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        intent = intents.get(predicted_class, "Unknown Intent")

    print(f"Classified intent: {intent} for text: '{text}'")  # Debugging line
    return intent

# Function to simulate backend integration for fetching real-time data (placeholder)
def fetch_from_backend(intent, user_input=None):
    # Simulated backend responses
    backend_responses = {
        "Order Status": "Your order is currently being processed and will be shipped soon.",
        "Product Inquiry": "We have several products in that category. Would you like to see our bestsellers?",
    }
    return backend_responses.get(intent, None)

# Function to generate response based on intent and conversation context
def generate_response(intent, user_id, context=None):
    responses = {
        "Product Inquiry": [
            "We have a wide range of products. Could you specify what you're looking for?",
            "Sure, I can help you with product details. Which product are you interested in?"
        ],
        "Order Status": [
            "To check the status of your order, could you please provide the order number?",
            "Please provide your order ID, and I'll check the status for you."
        ],
        "Return Request": [
            "To process your return request, please provide your order number and reason for return.",
            "I can assist with returns. Could you provide the order details?"
        ],
        "Payment Issues": [
            "Please describe the payment issue you're facing, and I'll do my best to assist.",
            "Can you provide more details about the payment problem? I'm here to help."
        ],
        "Shipping Information": [
            "For shipping details, please provide your order number.",
            "I'd be happy to help with shipping information. Do you have an order ID?"
        ],
        "Account Assistance": [
            "Could you tell me more about the issue with your account?",
            "I'm here to help with account issues. What specifically do you need assistance with?"
        ],
        "Cart Management": [
            "If you need help managing your cart, let me know what you'd like to change.",
            "I can help with cart management. Are there specific items you're having trouble with?"
        ]
    }

    # Fetch real-time data from backend if available
    backend_response = fetch_from_backend(intent)
    if backend_response:
        return backend_response

    # Choose a random response for variety
    response = random.choice(responses.get(intent, ["Sorry, I didn't understand that. Can you please clarify?"]))

    # Check if we need to follow up on any context
    if context and 'pending' in context and context['pending']:
        if intent == "Order Status" and 'order_id' not in context:
            response = "Could you please provide the order ID so I can check the status for you?"
            context['pending'] = True
        elif intent == "Return Request" and 'order_id' not in context:
            response = "I will need your order number to process the return request."
            context['pending'] = True
        else:
            context['pending'] = False

    print(f"Generated response: {response} for intent: '{intent}'")  # Debugging line
    return response

# Function to handle text input
def handle_text_input(text, user_id):
    try:
        context = conversation_history.get(user_id, {'pending': False})
        intent = classify_intent(text)
        response = generate_response(intent, user_id, context=context)
        conversation_history[user_id] = context
        return response
    except Exception as e:
        return f"Error processing text input: {e}"

# Function to handle audio input
def handle_audio_input(audio_bytes, user_id):
    try:
        recognizer = sr.Recognizer()
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        print(f"Recognized text from audio: '{text}'")  # Debugging line
        context = conversation_history.get(user_id, {'pending': False})
        intent = classify_intent(text)
        response_text = generate_response(intent, user_id, context=context)
        conversation_history[user_id] = context

        # Generate audio response
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            tts = gTTS(text=response_text, lang='en')
            tts.save(temp_audio_file.name)
            audio_file_path = temp_audio_file.name

        # Provide a link to download or play the audio response
        st.audio(audio_file_path, format='audio/mp3')
        os.remove(audio_file_path)  # Clean up the temporary file
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand the audio.")
    except sr.RequestError:
        st.error("Sorry, there was an error with the speech recognition service.")
    except Exception as e:
        st.error(f"Error processing audio input: {e}")

# Function to handle escalation to human agent
def escalate_to_human():
    st.write("It seems like your query requires more specialized attention. Transferring you to a human agent...")

# Streamlit interface
def main():
    st.title("E-commerce Customer Support System")

    user_id = st.text_input("Enter your User ID:", "guest")

    mode = st.selectbox("Choose Input Mode", ["Text", "Audio"])

    # Text input area
    st.subheader("Enter Your Query:")
    text_input = st.text_area("Enter your query:")
    if st.button("Submit", key="submit_text"):
        if text_input:
            response = handle_text_input(text_input, user_id)
            st.write("Response:", response)
            # Check if escalation is needed
            if response == "Sorry, I didn't understand that. Can you please clarify?":
                escalate_to_human()
        else:
            st.error("Please enter your query.")

    if mode == "Text":
        # Section for pre-built queries
        st.subheader("Pre-built Queries")
        query_buttons = {
            "Product Inquiry": "I'd like to know more about your products.",
            "Order Status": "What is the status of my order?",
            "Return Request": "How can I return a product?",
            "Payment Issues": "I have a problem with my payment.",
            "Shipping Information": "Where is my shipment?",
            "Account Assistance": "I need help with my account.",
            "Cart Management": "How can I manage my cart?"
        }

        for label, query in query_buttons.items():
            if st.button(label, key=f"query_button_{label}"):
                response = handle_text_input(query, user_id)
                st.write("Response:", response)

    elif mode == "Audio":
        st.subheader("Upload your audio file:")
        audio_file = st.file_uploader("Upload an audio file", type=["wav"])
        if st.button("Submit", key="submit_audio"):
            if audio_file is not None:
                handle_audio_input(audio_file.read(), user_id)
            else:
                st.error("Please upload an audio file.")

if __name__ == "__main__":
    main()
