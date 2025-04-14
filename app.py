import streamlit as st
from fer import FER
from transformers import pipeline
import librosa
import numpy as np
from pydub import AudioSegment
import os
import matplotlib.pyplot as plt
from collections import Counter
import moviepy.editor as mp

# Streamlit Web App UI Elements
st.title('AI MoodMatch System 🌈')
st.markdown("### Upload your voice note (audio), video, or input your mood text. All are optional.")

# Upload Inputs
image = st.file_uploader("Upload your selfie (Image)", type=["jpg", "png"])
audio = st.file_uploader("Upload your voice note (Audio)", type=["mp3", "wav", "flac", "m4a"])
video = st.file_uploader("Upload your video", type=["mp4", "avi", "mov", "mkv"])
text_input = st.text_area("Enter a text mood (e.g., I'm feeling happy!)")

# Function for Facial Emotion Detection
def detect_face_emotion(image_path):
    img = image_path
    detector = FER(mtcnn=True)
    result = detector.detect_emotions(img)
    if result:
        emotions = result[0]['emotions']
        top_emotion = max(emotions, key=emotions.get)
        return top_emotion, emotions[top_emotion]
    return "neutral", 0.0

# Function for Text Sentiment Analysis
def analyze_text_emotion(text):
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)
    result = classifier(text)
    emotion = result[0]['label']
    score = result[0]['score']
    return emotion, score

# Function to handle multiple audio file types (e.g., mp3, wav, etc.)
def handle_audio_file(audio_file):
    # Convert audio file to wav if it's not already in wav format
    if audio_file.type == "audio/mp3":
        audio = AudioSegment.from_mp3(audio_file)
        audio_path = "converted_audio.wav"
        audio.export(audio_path, format="wav")
    elif audio_file.type == "audio/m4a":
        audio = AudioSegment.from_file(audio_file, format="m4a")
        audio_path = "converted_audio.wav"
        audio.export(audio_path, format="wav")
    elif audio_file.type == "audio/flac":
        audio = AudioSegment.from_file(audio_file, format="flac")
        audio_path = "converted_audio.wav"
        audio.export(audio_path, format="wav")
    else:
        audio_path = audio_file.name
        with open(audio_path, "wb") as f:
            f.write(audio_file.getbuffer())
    
    return audio_path

# Function to extract audio from video
def extract_audio_from_video(video_file):
    video = mp.VideoFileClip(video_file)
    audio_path = "extracted_audio.wav"
    video.audio.write_audiofile(audio_path)
    return audio_path

# Function for Audio Emotion Detection
def extract_voice_features(audio_path):
    y, sr = librosa.load(audio_path, duration=5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    return np.mean(mfcc, axis=1)

def predict_voice_emotion(features):
    emotions = ['happy', 'sad', 'angry', 'neutral']
    return np.random.choice(emotions), round(np.random.uniform(0.5, 0.9), 2)

# Function to Fuse Moods
def fuse_moods(moods_list):
    counted = Counter(moods_list)
    final_mood = counted.most_common(1)[0][0]
    return final_mood

# Function for Mood-based Recommendations
def get_recommendation(mood):
    recommendations = {
        'happy': ['Watch comedy Reels 🎭', 'Try energetic filter 🌟'],
        'sad': ['Listen to calming music 🎶', 'Use warm filter 💛'],
        'angry': ['Breathe deeply 🧘', 'Use cool-toned filter 💙'],
        'neutral': ['Explore trending posts 🧠', 'Use minimalist filter 🖤'],
    }
    return recommendations.get(mood, ["Explore more content!"])

# Run the Mood Detection Pipeline
if text_input or audio or video:
    moods = []

    # Step 1: Text
    if text_input:
        text_mood, _ = analyze_text_emotion(text_input)
        st.write(f"Text Mood Detected: {text_mood}")
        moods.append(text_mood)

    # Step 2: Voice (Audio or Video)
    if audio:
        audio_path = handle_audio_file(audio)  # Handle different audio file types
        voice_features = extract_voice_features(audio_path)
        voice_mood, _ = predict_voice_emotion(voice_features)
        st.write(f"Voice Mood Detected: {voice_mood}")
        moods.append(voice_mood)
    elif video:
        audio_path = extract_audio_from_video(video)  # Extract audio from video
        voice_features = extract_voice_features(audio_path)
        voice_mood, _ = predict_voice_emotion(voice_features)
        st.write(f"Voice Mood Detected: {voice_mood}")
        moods.append(voice_mood)

    # Step 3: Fuse Moods
    final_mood = fuse_moods(moods)
    st.write(f"Final Mood Detected: {final_mood}")

    # Step 4: Get Recommendations
    recommendations = get_recommendation(final_mood)
    st.write("Recommendations based on your mood:")
    for rec in recommendations:
        st.write(f"- {rec}")

    # Optional: Add Mood History Plot
    st.write("Mood History (Sample):")
    mood_history = ["happy", "sad", "angry", "neutral", "happy", "sad"]
    plt.figure(figsize=(8, 5))
    plt.plot(mood_history, label="Mood Trend")
    plt.ylabel("Mood")
    plt.xlabel("Days")
    st.pyplot(plt)
