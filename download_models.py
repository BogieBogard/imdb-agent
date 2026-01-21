import os
import time
from huggingface_hub import snapshot_download
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def download_models():
    print("⬇️ Starting Model Download...")
    
    # 1. Download Whisper Model (The big one likely causing the hang)
    print("\n📦 Downloading OpenAI Whisper Large V3 Turbo...")
    try:
        snapshot_download(repo_id="openai/whisper-large-v3-turbo", repo_type="model")
        print("✅ Whisper Model Downloaded.")
    except Exception as e:
        print(f"❌ Failed to download Whisper: {e}")

    # 2. Download TTS Model
    print("\n📦 Downloading Microsoft SpeechT5 TTS...")
    try:
        snapshot_download(repo_id="microsoft/speecht5_tts", repo_type="model")
        snapshot_download(repo_id="microsoft/speecht5_hifigan", repo_type="model") # Vocoder often needed
        print("✅ TTS Model Downloaded.")
    except Exception as e:
        print(f"❌ Failed to download TTS: {e}")

    # 3. Download Embeddings (Used in agent.py)
    # Note: Ollama manages its own models, but if we used HF embeddings:
    # print("\n📦 Downloading Qwen Embeddings...")
    # snapshot_download(repo_id="Alibaba-NLP/gte-Qwen2-7B-instruct", repo_type="model") 

    print("\n🎉 Download process completed. Try running the app again.")

if __name__ == "__main__":
    # Optional: Set HF mirror if speed is an issue
    # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 
    download_models()
