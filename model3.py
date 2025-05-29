import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from io import BytesIO
import joblib

# Load trained model and label encoder
model = tf.keras.models.load_model("cry_classification_cts_model.h5")
le = joblib.load("label_encoder.pkl")

# Baby-friendly emoji dictionary using full class names
emoji_dict = {
    "belly_pain": "😢 - Belly Pain Cry",
    "burping": "😫 - Burping Cry",
    "discomfort": "😖 - Discomfort Cry",
    "cranky": "🥺 - Cranky Cry",
    "hungry": "🍽️ - Hungry Cry",
    "tired": "😴 - Tired Cry"
}

# Feature extractor
def extract_combined_features(file):
    y, sr = librosa.load(file, sr=16000)
    MAX_DURATION = 4
    MAX_LENGTH = sr * MAX_DURATION
    y = y[:MAX_LENGTH] if len(y) >= MAX_LENGTH else np.pad(y, (0, MAX_LENGTH - len(y)))

    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr), axis=1)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)

    return np.hstack([chroma, tonnetz, contrast])

# UI styling with background and fonts
background_url = "https://tse1.mm.bing.net/th?id=OIP.Z9jpBNQpUu3bC4uzvrPM-wHaE8&pid=Api"
st.markdown(f"""
    <style>
        .stApp {{
            background-image: url('{background_url}');
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            font-family: 'Comic Sans MS', cursive, sans-serif;
        }}
        .title-font {{
            font-size: 40px;
            color: #ff6f61;
            text-align: center;
            padding: 20px;
        }}
        .result-box {{
            background-color: rgba(255, 255, 255, 0.75);
            padding: 20px;
            border-radius: 20px;
            text-align: center;
            margin-top: 20px;
            border: 2px dashed #ffa07a;
        }}
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown('<h1 class="title-font">👶 Baby Cry Classifier 🍼</h1>', unsafe_allow_html=True)

# File upload section
st.subheader("🎵 Upload Baby's Cry (WAV file only)")
audio_file = st.file_uploader("Choose a file", type=["wav"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")

    try:
        # Feature extraction
        features = extract_combined_features(audio_file)
        features = features.reshape((1, 25, 1))

        # Predict
        pred = model.predict(features)
        label_index = np.argmax(pred)
        label = le.inverse_transform([label_index])[0]
        confidence = float(pred[0][label_index]) * 100

        # Show prediction
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown(f"### 👇 Predicted Cry")
        st.markdown(f"## {emoji_dict.get(label, '🔊 Unknown Cry')} ({confidence:.2f}% Confidence)")
        st.markdown('</div>', unsafe_allow_html=True)

        # Optional: Class probabilities
        st.markdown("#### 🔍 Prediction Probabilities:")
        for i, prob in enumerate(pred[0]):
            st.write(f"{le.classes_[i]}: {prob * 100:.2f}%")

    except Exception as e:
        st.error(f"❌ Something went wrong while processing the file: {e}")