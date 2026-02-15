

Speech-to-Text Sentiment Analyzer with Voice Response

This project converts user speech into text, performs sentiment/emotion analysis, and responds in spoken form using a cloned voice.

* **Speech-to-Text (STT):** Captures user speech via microphone
* **Emotion Analysis:** Uses a transformer-based classifier to detect emotions from text
* **Text-to-Speech (TTS):** Responds with a spoken summary of detected emotions
* **Voice Cloning:** Uses a reference audio for personalized speech output
* **Real-time Feedback:** Continuously listens and provides spoken responses

Requirements

```bash
pip install torch transformers TTS speechrecognition sounddevice soundfile
```

 Usage

1. Connect your microphone and set your cloned voice file path
2. Run the script
3. Speak naturally; the system will recognize your speech, detect emotion, and respond in voice

---

 Notes

* Uses Coqui TTS for multilingual speech synthesis
* Uses `j-hartmann/emotion-english-distilroberta-base` for emotion classification
* Works on CPU or GPU (CUDA) for TTS
* Continuously listens for new input until stopped

