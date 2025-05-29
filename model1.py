import streamlit as st
import numpy as np
import librosa
import cv2
import tensorflow as tf
import pickle

# Page config
st.set_page_config(
    page_title="Baby Cry Detector 👶🔊",
    page_icon="🍼",
    layout="centered"
)

# Custom CSS for baby-themed pattern background
st.markdown("""
    <style>
        body {
            background: url('https://www.toptal.com/designers/subtlepatterns/uploads/baby-pattern.png');  /* subtle pattern */
            background-color: #fff7f0; /* fallback peach */
            background-size: auto;
        }
        .main {
            background-color: rgba(255, 247, 240, 0.95); /* slightly transparent peach */
            padding: 2em;
            border-radius: 10px;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #ff6f61;
            text-align: center;
        }
        .subtitle {
            font-size: 18px;
            color: #555;
            text-align: center;
        }
        .stButton button {
            background-color: #ff6f61;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
            font-size: 16px;
        }
        .stButton button:hover {
            background-color: #ff4b3e;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">👶 Baby Cry Classification With Mel Spectrogram</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a `.wav` file and I’ll detect what kind of cry it is</div><br>', unsafe_allow_html=True)

# Load the model and label encoder
model = tf.keras.models.load_model("cry_classification_mel_model.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Constants
SAMPLE_RATE = 16000
N_MELS = 128
MAX_DURATION = 4  # seconds
MAX_LENGTH = SAMPLE_RATE * MAX_DURATION
FIXED_SIZE = (128, 128)

# Cry type suggestions
cure_map = {
    'hungry': "🍼 **Feed your baby**. They might be asking for milk or food!",
    'discomfort': "😖 **Check the diaper, temperature, or tight clothing**.",
    'tired': "😴 **Rock them gently to sleep** or place in a dark, quiet room.",
    'burping': "🤢 **Hold the baby upright and gently pat their back**.",
    'belly_pain': "🤰 **Gently massage the tummy** or try cycling the legs to relieve gas."
}

emoji_map = {
    'hungry': '🍽️',
    'pain': '💢',
    'discomfort': '😖',
    'tired': '😴',
    'burping': '🤢',
    'belly_pain': '🤰'
}

# Audio feature extraction
def extract_mel(file, sr=SAMPLE_RATE, n_mels=N_MELS, fixed_size=FIXED_SIZE):
    y, _ = librosa.load(file, sr=sr)
    y = y / np.max(np.abs(y))  # Normalize
    y, _ = librosa.effects.trim(y)  # Trim silence

    # Pad or truncate
    if len(y) < MAX_LENGTH:
        y = np.pad(y, (0, MAX_LENGTH - len(y)))
    else:
        y = y[:MAX_LENGTH]

    # Convert to Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_resized = cv2.resize(mel_db, fixed_size)
    mel_resized = mel_resized[..., np.newaxis]

    return mel_resized

# File uploader
uploaded_file = st.file_uploader("📁 Upload a baby cry audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    st.markdown("---")

    with st.spinner("🧠 Analyzing cry pattern..."):
        features = extract_mel(uploaded_file)
        features = np.expand_dims(features, axis=0)
        prediction = model.predict(features)
        predicted_label = le.inverse_transform([np.argmax(prediction)])[0]

        # Output predicted label
        cry_type = predicted_label.lower()
        emoji = emoji_map.get(cry_type, "👶")
        st.success(f"🔍 **Predicted Cry Type: `{predicted_label.upper()}`** {emoji}")

        # Suggest cure
        st.markdown("### 💡 Suggested Cure")
        st.info(cure_map.get(cry_type, "Try soothing with gentle cuddles or lullabies."))

    st.markdown("---")
    st.caption("📌 Tip: Try uploading different baby cries to explore the predictions and remedies!")
