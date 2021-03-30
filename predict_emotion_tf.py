#!/usr/bin/env python3

import speech_recognition as sr
import os

import gtts
from pydub import AudioSegment
from pydub.playback import play

import numpy as np
import pickle

from record_audio import get_audio

import scipy

import tensorflow as tf

import tensorflow_text as text

import tensorflow_hub as hub

import librosa

# Import TF 2.X and make sure we're running eager.
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
assert tf.executing_eagerly()

import warnings
warnings.filterwarnings('ignore')


# Constants
DURATION_IEMOCAP = 11
SAMPLING_RATE = 16000
input_length = SAMPLING_RATE * DURATION_IEMOCAP
# MER model
# MER_ELECTRA_TRILL = '/content/mser-thesis-app/result_models/mer_trill_electra_small_model.h5'
MER_ELECTRA_TRILL = './mser-thesis-app/result_models/mer_trill_electra_small_model.h5'
# Emotion models
emotions_iemocap = ['neutral', 'happy', 'sad', 'angry']
emotions_ravdess = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
# Definition of language
LANG='en'
# BERT model types
map_name_to_handle = {
    'electra_small':
        'https://tfhub.dev/google/electra_small/2',
}
map_model_to_preprocess = {
    'electra_small':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
}


"""
    This function predicts an emotion of the given audio file with corresponding transript.
"""
def predict_emotion(text, filepath, emotions=emotions_iemocap):
    # Load the module and run inference.
    trill_module = hub.load('https://tfhub.dev/google/nonsemantic-speech-benchmark/trill-distilled/3')
    
    y, _ = librosa.load(filepath, sr=SAMPLING_RATE)
    # y,_ = librosa.effects.trim(y, top_db = 25)
    # https://en.wikipedia.org/wiki/Wiener_filter
    # https://cs.wikipedia.org/wiki/Wiener%C5%AFv_filtr
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.wiener.html
    y = scipy.signal.wiener(y)

    if len(y) > input_length:
        # Cut to the same length 
        y = y[0:input_length]
    elif input_length > len(y):
        # Pad the sequence
        max_offset = input_length - len(y)  
        y = np.pad(y, (0, max_offset), "constant")

    X_audio = np.array([trill_module(samples=y, sample_rate=SAMPLING_RATE)['embedding'].numpy()])

    X_text = np.array([text])

    # Load the model
    model = tf.keras.models.load_model(MER_ELECTRA_TRILL, custom_objects={'KerasLayer':hub.KerasLayer})

    # Compile the model
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(3e-5),
                metrics=['accuracy'])
    
    # Get the predicted emotion index
    pred_id = tf.argmax(model.predict([X_text, X_audio]), 1).numpy()[0]

    return emotions[pred_id]


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


