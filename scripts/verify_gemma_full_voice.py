
import requests
import base64
import json
import os
import wave

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:12b"

def create_dummy_wav(filename="test.wav"):
    with wave.open(filename, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b'\x00' * 16000) # 1 sec silence
    return filename

def test_stt():
    print("Testing STT (Audio -> Text)...")
    create_dummy_wav()
    
    with open("test.wav", "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode('utf-8')
        
    payload = {
        "model": MODEL,
        "prompt": "Transcribe this audio file.",
        # effective field for audio in early 2026 ollama might be 'audio' or 'images'
        # Trying 'images' as placeholder, if fails might need 'audio'
        "images": [audio_b64] 
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=False)
        if response.status_code == 200:
            res_text = response.json().get("response", "")
            print(f"STT Response: {res_text}")
        else:
            print(f"STT Failed: {response.text}")
    except Exception as e:
        print(f"STT Exception: {e}")

def test_tts():
    print("\nTesting TTS (Text -> Audio)...")
    payload = {
        "model": MODEL,
        "prompt": "Generate audio saying: Hello World"
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=False)
        if response.status_code == 200:
            # Check for audio field or base64 in response
            # Note: Ollama might return 'response' as text. Audio might be separate?
            # Creating hypothesis that it returns text. Multimodal output usually separate.
            print(f"TTS Response Body Keys: {response.json().keys()}")
            print(f"TTS Response Text: {response.json().get('response')}")
        else:
            print(f"TTS Failed: {response.text}")
    except Exception as e:
        print(f"TTS Exception: {e}")

if __name__ == "__main__":
    test_stt()
    test_tts()
