import os
import numpy as np
import librosa
import speech_recognition as sr
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from pydub import AudioSegment

def list_microphones():
    print("Available microphones:")
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"{index}: {name}")

def record_audio(device_index=None, duration=30):
    recognizer = sr.Recognizer()
    with sr.Microphone(device_index=device_index) as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)  # 잡음 제거를 위한 조정
        audio = recognizer.listen(source, timeout=duration)
        print("Recording complete")
        return audio

def save_audio_to_file(audio, file_path):
    with open(file_path, "wb") as file:
        file.write(audio.get_wav_data())

def convert_m4a_to_wav(m4a_file_path, wav_file_path):
    audio = AudioSegment.from_file(m4a_file_path, format="m4a")
    audio.export(wav_file_path, format="wav")

def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def train_classifier():
    # 예제 데이터를 사용하여 분류기를 훈련합니다.
    # 실제 사용 시, 감정 레이블이 있는 음성 데이터를 사용하세요.
    X = []
    y = []

    # 긍정적인 예제
    pos_files = ['positive1.wav', 'positive2.wav']  # 예제 파일 경로
    for file in pos_files:
        features = extract_features(file)
        X.append(features)
        y.append('positive')

    # 부정적인 예제
    neg_files = ['negative1.wav', 'negative2.wav']  # 예제 파일 경로
    for file in neg_files:
        features = extract_features(file)
        X.append(features)
        y.append('negative')

    X = np.array(X)
    y = np.array(y)

    # SVM 분류기 훈련
    classifier = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
    classifier.fit(X, y)

    return classifier

def analyze_emotion_from_audio(file_path, classifier):
    features = extract_features(file_path)
    prediction = classifier.predict([features])
    return prediction[0]

def main():
    list_microphones()
    try:
        device_index = int(input("Select microphone device index: "))
    except ValueError:
        print("Invalid input. Using default microphone.")
        device_index = None

    audio = record_audio(device_index, duration=30)  # 녹음 시간을 30초로 설정
    m4a_file_path = r"C:\Users\phot1\theVoice\project\Positive.m4a"
    wav_file_path = r"C:\Users\phot1\theVoice\project\Positive.wav"

    # save_audio_to_file(audio, m4a_file_path)
    convert_m4a_to_wav(m4a_file_path, wav_file_path)

    m4a_file_path = r"C:\Users\phot1\theVoice\project\Negative.m4a"
    wav_file_path = r"C:\Users\phot1\theVoice\project\Negative.wav"
    
    # save_audio_to_file(audio, m4a_file_path)

    # m4a 파일을 wav 파일로 변환
    convert_m4a_to_wav(m4a_file_path, wav_file_path)

    classifier = train_classifier()
    emotion = analyze_emotion_from_audio(wav_file_path, classifier)
    if emotion:
        print(f"Detected Emotion: {emotion}")
    else:
        print("Could not analyze emotions.")

if __name__ == "__main__":
    main()
