import torch
from TTS.api import TTS
import speech_recognition as sr
from transformers import pipeline
import os
import sounddevice as sd
import soundfile as sf


r = sr.Recognizer()

# Load Coqui TTS
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Load emotion classifier
classifier = pipeline("text-classification",
                      model="j-hartmann/emotion-english-distilroberta-base",
                      top_k=None)


speaker_wav_path = ""

# Function to speak text using cloned voice
def SpeakWithClonedVoice(text):
    output_path = ""
    tts.tts_to_file(text=text, speaker_wav=speaker_wav_path, language="en", file_path=output_path)
    data, samplerate = sf.read(output_path)
    sd.play(data, samplerate)
    sd.wait()

while True:
    try:
        with sr.Microphone() as source2:
            r.adjust_for_ambient_noise(source2, duration=0.5)
            print(sd.query_devices())

            print("Listening...")
            audio2 = r.listen(source2)
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()

            print("Did you say:", MyText)
            SpeakWithClonedVoice("You said: " + MyText)

            emotions = classifier(MyText)[0]
            print("Detected emotions:")
            for emotion in emotions:
                print(f"{emotion['label']}: {round(emotion['score'] * 100, 2)}%")

            top_emotion = max(emotions, key=lambda x: x['score'])
            SpeakWithClonedVoice(f"I think you're feeling {top_emotion['label']}")

    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        SpeakWithClonedVoice("Sorry, I could not reach the speech service.")

    except sr.UnknownValueError:
        print("Unknown error occurred")
        SpeakWithClonedVoice("Sorry, I did not catch that.")
