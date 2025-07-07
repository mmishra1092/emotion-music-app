import os
import cv2
import numpy as np
import pandas as pd
import librosa
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from transformers import BertTokenizer, BertForSequenceClassification
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit as st

def load_fer2013(data_dir):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    emotion_to_label = {emotion: idx for idx, emotion in enumerate(emotions)}
    X, y = [], []
    for subset in ['train', 'test']:
        for emotion in emotions:
            folder = os.path.join(data_dir, subset, emotion)
            for img_file in os.listdir(folder):
                if img_file.endswith('.jpg'):
                    img = cv2.imread(os.path.join(folder, img_file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (48, 48))
                        X.append(img)
                        y.append(emotion_to_label[emotion])
    X = np.array(X).reshape(-1, 48, 48, 1) / 255.0
    y = np.array(y)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def get_image_emotion(image, model):
    img = Image.fromarray(image.astype('uint8')).convert('L').resize((48, 48))
    img = np.expand_dims(np.array(img), axis=(0, -1)) / 255.0
    return np.argmax(model.predict(img))

def get_text_emotion(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    if torch.cuda.is_available():
        model.to('cuda')
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=1).item()

def get_audio_emotion(audio_path, model):
    mfcc = extract_mfcc(audio_path).reshape(1, 1, -1)
    return np.argmax(model.predict(mfcc))

def fused_emotion(image, text, audio_path, image_model, text_model, tokenizer, audio_model):
    preds = [
        get_image_emotion(image, image_model),
        get_text_emotion(text, text_model, tokenizer),
        get_audio_emotion(audio_path, audio_model)
    ]
    return max(set(preds), key=preds.count)

emotion_to_music = {
    0: {"mood": "chill", "valence": 0.2},
    1: {"mood": "pop", "valence": 0.8},
    2: {"mood": "rock", "valence": 0.1},          # was metal → now valid
    3: {"mood": "ambient", "valence": 0.1},
    4: {"mood": "electronic", "valence": 0.6},
    5: {"mood": "acoustic", "valence": 0.5},
    6: {"mood": "blues", "valence": 0.3}
}

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="8a0d569b187a42a0a88de5fd710b0f36", client_secret="53f9fab414e846ab95bd9347de5a83c8"))

def recommend_music(emotion_label):
    mood = emotion_to_music[emotion_label]['mood']
    valence = emotion_to_music[emotion_label]['valence']
    try:
        results = sp.recommendations(seed_genres=[mood], target_valence=valence, limit=5)
        return [f"{t['name']} by {t['artists'][0]['name']}: {t['external_urls']['spotify']}" for t in results['tracks']]
    except Exception as e:
        return [f"Error fetching recommendations: {e}"]

model_fer = load_model("fer_model.h5")
model_audio = load_model("best_audio_lstm_model.h5")
model_text = BertForSequenceClassification.from_pretrained("./saved_model/bert_goemotions")
tokenizer = BertTokenizer.from_pretrained("./saved_model/bert_goemotions")

st.set_page_config(page_title="Emotion-Based Music Recommender", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🎵 Emotion-Aware Music Recommendation System</h1>", unsafe_allow_html=True)

st.markdown("Upload your **facial image**, enter your **text input**, and **upload a voice clip** to get personalized music recommendations based on your emotion!")

col1, col2, col3 = st.columns(3)

with col1:
    image_file = st.file_uploader("📷 Facial Image", type=["jpg", "jpeg", "png"])

with col2:
    text_input = st.text_input("💬 Text Input", placeholder="How are you feeling today?")

with col3:
    audio_file = st.file_uploader("🎤 Audio File (WAV)", type=["wav"])

if st.button("🎯 Detect Emotion & Recommend Music"):
    if image_file and text_input and audio_file:
        with st.spinner("🔍 Detecting your emotion..."):
            try:
                image = Image.open(image_file).convert('L')
                image_np = np.array(image)
                emotion = fused_emotion(image_np, text_input, audio_file.name, model_fer, model_text, tokenizer, model_audio)
                mood = emotion_to_music[emotion]['mood']
                valence = emotion_to_music[emotion]['valence']

                st.success(f"🧠 Detected Emotion: **{mood.capitalize()}** (Valence: {valence})")

                st.markdown("---")
                st.subheader("🎧 Recommended Tracks")

                recommendations = recommend_music(emotion)
                for i, track in enumerate(recommendations, 1):
                    st.markdown(f"{i}. {track}")

            except Exception as e:
                st.error(f"❌ An error occurred during processing:\n{e}")
    else:
        st.warning("⚠️ Please provide **all three inputs**: image, text, and audio.")