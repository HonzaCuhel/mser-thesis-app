#!/usr/bin/env python3

import speech_recognition as sr
import os

import gtts
from pydub import AudioSegment
from pydub.playback import play

import numpy as np
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from extract_audio_features import extract_audio_features
from record_audio import get_audio


# Constants
DURATION_IEMOCAP = 11
SAMPLING_RATE = 16000
LR_MER_MODEL = '/content/mser-thesis-app/result_models/lr_mer_iemocap'
VECTORIZER = '/content/mser-thesis-app/mer_tfidf_iemocap.pkl'
SCALER = '/content/mser-thesis-app/mer_mfcc_mel_chroma_scaler_iemocap.pkl'

emotions = ['neutral', 'happy', 'sad', 'angry']
 
LANG='en'
 
vectorizer_params = {
    'ngram_range': (1, 2),
    'max_df': 0.95,
    'sublinear_tf': True,
    'min_df': 4,
    'stop_words': 'english',
    'max_features': 2200
}

LR_PARAMS = {
    'solver': 'newton-cg',
    'penalty': 'l2',
    'multi_class': 'auto',
    'max_iter': 800,
    'class_weight': None,
    'C': 0.56
}


"""
    This function predicts an emotion of the given audio file with corresponding transript.
"""
def predict_emotion(text, filepath, saved_model_filename=LR_MER_MODEL, scaler_filename=SCALER, vect_filename=VECTORIZER):
    # Get the features
    X_audio = extract_audio_features(filepath, mfcc=True, mel=True, chroma=True).reshape(1, -1)

    # Create a normalization object
    scaler = MinMaxScaler()
    # Load the scaler
    with open(scaler_filename, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    # Normalize audio training data
    X_audio = scaler.transform(X_audio)

    # Initialize Vectorizer
    vectorizer = TfidfVectorizer(**vectorizer_params)
    # Load the vectorizer
    with open(vect_filename, 'rb') as vect_file:
        vectorizer = pickle.load(vect_file)

    # Get the tf-idf text features
    X_text = vectorizer.transform([text]).toarray()

    # X = np.hstack(([text], X_audio[0])).reshape(1, -1)
    X = np.concatenate((X_text, X_audio), axis=1)

    model = LogisticRegression(**LR_PARAMS)
    # Load the model
    with open(saved_model_filename, 'rb') as f:
        model = pickle.load(f)
    
    return emotions[model.predict(X)[0]]


"""
    This method runs the EmotionRecognition.
"""
def run_emotion_recognizer():
    # Record an audio from a microphone
    audio, sample_rate, audio_file = get_audio()

    # use the audio file as the audio source
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source, duration=DURATION_IEMOCAP)  # read the entire audio file

    # Resource: https://github.com/Uberi/speech_recognition/blob/master/examples/audio_transcribe.py
    # Recognize speech using Google Speech Recognition
    try:
        text = r.recognize_google(audio, language=lang)
        print("Google Speech Recognition thinks you said: " + text)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    
    # Predict an emotion
    pred = predict_emotion(text, audio_file)

    print('\n' + '-'*40)
    print(f'Predicted emotion: {pred}')

    # make request to google to get synthesis
    tts = gtts.gTTS(f'You are {pred}', lang=LANG)

    output_file = 'output_emotion.mp3'
    tts.save(output_file)

    sound = AudioSegment.from_mp3(output_file)
    play(sound)

    # Delete the files
    os.remove(filepath)
    os.remove(output_file)


