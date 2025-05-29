import streamlit as st
import numpy as np
import librosa
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
            background: url('https://www.toptal.com/designers/subtlepatterns/uploads/baby-pattern.png');
            background-color: #fff7f0;
            background-size: auto;
        }
        .main {
            background-color: rgba(255, 247, 240, 0.95);
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
st.markdown('<div class="title">👶 Baby Cry Classification With MFCC</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a `.wav` file and I’ll detect what kind of cry it is</div><br>', unsafe_allow_html=True)

# Load model and label encoder
model = tf.keras.models.load_model("cry_classification_mfcc_model.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Constants
SAMPLE_RATE = 16000
NUM_MFCC = 40
MAX_DURATION = 4  # seconds
MAX_LENGTH = SAMPLE_RATE * MAX_DURATION

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

# Extract MFCC features with fixed shape
def extract_mfcc(file, sr=SAMPLE_RATE, num_mfcc=NUM_MFCC, max_duration=MAX_DURATION):
    y, _ = librosa.load(file, sr=sr)

    # Pad or trim
    if len(y) < sr * max_duration:
        y = np.pad(y, (0, sr * max_duration - len(y)))
    else:
        y = y[:sr * max_duration]

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)

    # Normalize
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

    # Make sure MFCC has shape (100, 40)
    if mfcc.shape[1] != 100:
        mfcc = librosa.util.fix_length(mfcc, size=100, axis=1)

    mfcc = mfcc.T  # shape becomes (100, 40)
    return mfcc

# File uploader
uploaded_file = st.file_uploader("📁 Upload a baby cry audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    st.markdown("---")

    try:
        # Extract features
        features = extract_mfcc(uploaded_file)  # shape: (100, 40)

        # Expand dims: (100, 40) → (1, 100, 40, 1)
        features = np.expand_dims(features, axis=0)
        features = np.expand_dims(features, axis=-1)

        # Predict
        with st.spinner("🧠 Analyzing cry pattern..."):
            prediction = model.predict(features)
            predicted_label = le.inverse_transform([np.argmax(prediction)])[0]

            # Show result
            cry_type = predicted_label.lower()
            emoji = emoji_map.get(cry_type, "👶")
            st.success(f"🔍 **Predicted Cry Type: `{predicted_label.upper()}`** {emoji}")

            st.markdown("### 💡 Suggested Cure")
            st.info(cure_map.get(cry_type, "Try soothing with gentle cuddles or lullabies."))

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

    st.markdown("---")
    st.caption("📌 Tip: Try uploading different baby cries to explore the predictions and remedies!")
