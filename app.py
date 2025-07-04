import streamlit as st
import pickle
import numpy as np
import time
import requests

# Load animation from Lottie
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load trained model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Page setup
st.set_page_config(page_title="Mental Health Predictor", page_icon="🧠", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #fdfbfb, #ebedee);
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .css-1cpxqw2 {
        border-radius: 12px;
        padding: 10px;
        background: #ffffff88;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.05);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .positive {
        background: linear-gradient(135deg, #4caf50, #81c784);
    }
    .negative {
        background: linear-gradient(135deg, #e53935, #ef5350);
    }
    footer {
        color: #aaa;
        text-align: center;
        margin-top: 40px;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# Title & Animation
st.markdown("<h1 style='text-align: center;'>🧠 Mental Health Prediction</h1>", unsafe_allow_html=True)

# ✅ Replacing st_lottie with HTML-based Lottie animation
st.components.v1.html("""
    <div style='display: flex; justify-content: center;'>
        <lottie-player src="https://lottie.host/5299d1bb-6a17-4cc1-bd16-fd9897641f63/fWeCHLNeCy.json"
                       background="transparent"
                       speed="1"
                       style="width: 300px; height: 300px;"
                       loop autoplay></lottie-player>
    </div>
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
""", height=300)

st.markdown("---")

# Input form
st.header("📋 Fill the Form Below")
age = st.slider("Age", 18, 60, 25)
family_history = st.selectbox("Do you have a family history of mental illness?", ["No", "Yes"])
care_options = st.selectbox("Do you have access to mental health care at work?", ["No", "Yes"])
work_interfere = st.selectbox("How often does your work interfere with mental health?", ["Never", "Rarely", "Sometimes", "Often"])
obs_consequence = st.selectbox("Have you observed mental health consequences at work?", ["No", "Yes"])
benefits = st.selectbox("Does your employer provide mental health benefits?", ["No", "Yes"])
mental_health_consequence = st.selectbox("Would mental health affect your work decision?", ["No", "Maybe", "Yes"])
gender = st.selectbox("Gender", ["Female", "Male", "Other"])
mental_vs_physical = st.selectbox("Do you value mental health equal to physical?", ["No", "Yes"])

# Prepare input
input_dict = {
    'Age': age,
    'family_history': 1 if family_history == "Yes" else 0,
    'care_options_Yes': 1 if care_options == "Yes" else 0,
    'work_interfere_Often': 1 if work_interfere == "Often" else 0,
    'obs_consequence_Yes': 1 if obs_consequence == "Yes" else 0,
    'benefits_Yes': 1 if benefits == "Yes" else 0,
    'work_interfere_Rarely': 1 if work_interfere == "Rarely" else 0,
    'mental_health_consequence_Maybe': 1 if mental_health_consequence == "Maybe" else 0,
    'Gender_Male': 1 if gender == "Male" else 0,
    'mental_vs_physical_Yes': 1 if mental_vs_physical == "Yes" else 0
}

input_array = np.array(list(input_dict.values())).reshape(1, -1)

# Predict
if st.button("Predict", use_container_width=True):
    with st.spinner("🧠 Processing your mental health status..."):
        time.sleep(1.5)
        scaled_input = scaler.transform(input_array)
        result = model.predict(scaled_input)[0]

    # Display result
    if result == 0:
        st.markdown("<div class='prediction-box positive'>🟢 You likely do NOT need mental health treatment. Keep taking care of yourself!</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='prediction-box negative'>🔴 You MAY need mental health support. It's okay to ask for help 💛</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <footer>
    Made with 💙 for awareness and support<br>
    <b>Bhavani</b> 🧠
    </footer>
""", unsafe_allow_html=True)
