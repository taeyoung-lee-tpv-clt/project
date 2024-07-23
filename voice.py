import speech_recognition as sr
from transformers import pipeline

def list_microphones():
    print("Available microphones:")
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"{index}: {name}")

def speech_to_text(device_index=None):
    recognizer = sr.Recognizer()
    with sr.Microphone(device_index=device_index) as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            print("Recognizing...")
            text = recognizer.recognize_google(audio, language='en-US')
            print(f"Recognized Text: {text}")
            return text
        except Exception as e:
            print(f"Error: {e}")
            return None

def analyze_sentiment(text):
    # 명시적으로 모델과 버전을 지정합니다.
    model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
    sentiment_analyzer = pipeline('sentiment-analysis', model=model_name)
    result = sentiment_analyzer(text)
    return result

def main():
    list_microphones()
    device_index = int(input("Select microphone device index: "))
    text = speech_to_text(device_index)
    if text:
        sentiment = analyze_sentiment(text)
        print(f"Sentiment: {sentiment}")

if __name__ == "__main__":
    main()