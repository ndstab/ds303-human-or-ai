import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from preprocess import clean_text, tokenize, encode_tokens, pad_sequence

# Set page config with custom theme
st.set_page_config(
    page_title="DS303 : Human VS AI",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for futuristic dark theme
st.markdown("""
    <style>
    /* Import Open Sans font */
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;500;600&display=swap');
    
    /* Apply font to all elements */
    * {
        font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif !important;
    }
    
    /* Main background */
    .stApp {
        background-color: #0a0a1a;
    }
    
    /* Text area styling */
    .stTextArea>div>div>textarea {
        background-color: #1a1a2e;
        color: #00ffff;
        border: 1px solid #1e3a8a;
        font-family: 'Open Sans', sans-serif !important;
        min-height: 150px !important;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #1e3a8a;
        color: #00ffff;
        border: none;
        padding: 10px 25px;
        font-size: 1.1em;
        border-radius: 5px;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 10px;
        font-weight: 600;
    }
    
    .stButton>button:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6 {
        color: #00ffff !important;
    }
    
    p, div {
        color: #e0e0ff !important;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #1a1a2e;
        color: #00ffff !important;
        border: 1px solid #1e3a8a;
        padding: 20px;
        border-radius: 5px;
        margin: 20px 0;
    }
    
    /* Warning message styling */
    .stWarning {
        background-color: #1a1a2e;
        color: #ff3e96 !important;
        border: 1px solid #ff3e96;
        padding: 20px;
        border-radius: 5px;
        margin: 20px 0;
    }
    
    /* Info message styling */
    .stInfo {
        background-color: #1a1a2e;
        color: #00ffff !important;
        border: 1px solid #1e3a8a;
        padding: 20px;
        border-radius: 5px;
        margin: 20px 0;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #1e3a8a;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #2563eb;
    }

    /* Footer styling */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        width: 100%;
        background-color: #0a0a1a;
        padding: 8px;
        text-align: center;
        border-top: 1px solid #1e3a8a;
    }
    
    .footer p {
        margin: 0;
        padding: 2px;
        font-size: 0.9em;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model and vocab
@st.cache_resource
def load_all():
    model = load_model("model/lstm_ai_human_classifier.h5")
    with open("model/word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)
    return model, word2idx

model, word2idx = load_all()

# Prediction function
def predict(text):
    tokens = tokenize(clean_text(text))
    encoded = encode_tokens(tokens, word2idx)
    padded = pad_sequence(encoded, word2idx)
    padded = np.array([padded])
    prob = float(model.predict(padded)[0][0])  # Convert to float to avoid numpy type
    label = "AI" if prob > 0.5 else "Human"
    # Ensure confidence is rounded to 2 decimal places
    confidence = float(round(prob * 100, 2)) if label == "AI" else float(round((1 - prob) * 100, 2))
    return label, confidence

# Title with custom styling
st.markdown("<h1 style='text-align: center; font-size: 2.2em;'>DS303 : AI or Human?</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #1e3a8a !important; font-size: 1.3em;'>Detect whether the text was written by an AI or a human</h3>", unsafe_allow_html=True)

# Add some space
st.write("")

# Information box
st.info("📝 Paste any text below and our model will analyze whether it was written by AI or a human. For best results, use texts above 300 words.")

# Text input area with custom styling
user_input = st.text_area(
    "Input Text:",
    placeholder="Enter your text here...",
    height=150
)

# Center the classify button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    classify_button = st.button("🔍 Analyze Text")

if classify_button:
    if user_input.strip():
        # Add a spinner while processing
        with st.spinner("Analyzing text..."):
            label, confidence = predict(user_input)
            # Create a more detailed result display
            st.markdown("### 📊 Analysis Results")
            
            # Add a container with ID for scrolling
            st.markdown('<div id="results"></div>', unsafe_allow_html=True)
            
            # Display the result with custom styling
            result_color = "#00ffff" if label == "Human" else "#ff3e96"
            st.markdown(f"""
                <div style='background-color: #1a1a2e; padding: 15px; border-radius: 8px; border: 1px solid {result_color};'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <h4 style='color: {result_color} !important; margin: 0;'>Classification: {label}</h4>
                        <span style='color: {result_color}; font-weight: 600;'>{confidence:.2f}%</span>
                    </div>
                    <div style='background-color: #0a0a1a; height: 6px; border-radius: 3px; margin-top: 8px;'>
                        <div style='background-color: {result_color}; width: {confidence:.2f}%; height: 100%; border-radius: 3px;'></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Add JavaScript for auto-scrolling
            st.markdown("""
                <script>
                    window.onload = function() {
                        document.getElementById('results').scrollIntoView({
                            behavior: 'smooth',
                            block: 'start'
                        });
                    }
                </script>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text to analyze!")

# Footer
st.markdown("""
    <div class='footer'>
        <p style='color: #00ffff;'>AI Text Detection Model</p>
        <p style='color: #e0e0ff;'>Aryan Badkul - 23B0689 | Sajjad Nakhwa - 23B0702 | Puranjay Bansal - 23B0731 | Shourya Goyal - 23B0733 | Rishabh Agarwal - 23B0758</p>
    </div>
    """, unsafe_allow_html=True)
