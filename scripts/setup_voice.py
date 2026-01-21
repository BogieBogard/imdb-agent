
import os
import requests

KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json"

MODEL_DIR = "models"

def download_file(url, output_path):
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        return
    
    print(f"Downloading {url} to {output_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download: {e}")

def main():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    download_file(KOKORO_MODEL_URL, os.path.join(MODEL_DIR, "kokoro-v0_19.onnx"))
    download_file(VOICES_URL, os.path.join(MODEL_DIR, "voices.json"))
    
    print("Kokoro model setup complete.")

if __name__ == "__main__":
    main()
