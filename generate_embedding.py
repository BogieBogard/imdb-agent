import torch
import torchaudio
import os

# Monkey-patch: SpeechBrain 1.0.3 uses 'list_audio_backends' which was removed in recent torchaudio versions.
if not hasattr(torchaudio, "list_audio_backends"):
    def _list_audio_backends():
        return ["soundfile"] # Assume soundfile is available (we installed it)
    torchaudio.list_audio_backends = _list_audio_backends

from speechbrain.inference.speaker import EncoderClassifier

def generate_embedding():
    print("🎤 Generating trusted reference audio using macOS 'say'...")
    ref_audio_path = "ref_audio.wav"
    
    # Use macOS system TTS to generate a clean, trusted human-like voice sample
    # -v 'Samantha' is a high-quality default voice on many Macs, but we'll let it pick default if missing.
    # We use a phonetically rich sentence for better embedding.
    text = "The quick brown fox jumps over the lazy dog. This is a trusted local audio reference."
    ret = os.system(f"say -v Samantha -o {ref_audio_path} --data-format=LEF32@16000 '{text}'")
    
    if ret != 0:
        # Fallback if 'Samantha' not found or other error, try default voice
        print("⚠️ 'Samantha' voice not found or failed, trying default voice...")
        ret = os.system(f"say -o {ref_audio_path} --data-format=LEF32@16000 '{text}'")
        
    if not os.path.exists(ref_audio_path):
        print("❌ Failed to create reference audio. Ensure you are on macOS or provide a 'ref_audio.wav' manually.")
        return

    print("🧠 Loading SpeechBrain Speaker Recognition Model...")
    # This downloads the trusted model from verify speechbrain organization
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="tmp_model")
    
    print("🔢 Computing Speaker Embedding...")
    import soundfile as sf
    # Bypass torchaudio.load completely due to environment issues
    data, fs = sf.read(ref_audio_path)
    # Convert to tensor. Soundfile returns (time,) for mono or (time, channels)
    signal = torch.from_numpy(data).float()
    
    # Ensure (1, time) format like torchaudio
    if len(signal.shape) == 1:
        signal = signal.unsqueeze(0) # (time) -> (1, time)
    else:
        signal = signal.t() # (time, ch) -> (ch, time)
        
    embeddings = classifier.encode_batch(signal)
    
    # The model outputs (batch, 1, 512), we need (1, 512)
    # SpeechT5 expects (1, 512)
    embedding_tensor = embeddings.squeeze(1)
    
    output_path = "speaker_embedding.pt"
    torch.save(embedding_tensor, output_path)
    
    print(f"✅ Success! Saved secure speaker embedding to '{output_path}'")
    print(f"Tensor shape: {embedding_tensor.shape}")
    
    # Cleanup
    if os.path.exists(ref_audio_path):
        os.remove(ref_audio_path)
    # Optional: cleanup tmp_model dir if desired, but keeping it caches the model

if __name__ == "__main__":
    generate_embedding()
