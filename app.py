import streamlit as st
import os
from agent import MovieAgent
# import langchain
# langchain.debug = True

st.set_page_config(page_title="IMDB Voice Agent", page_icon="🎬")

st.title("🎬 IMDB Movie Voice Agent")
st.markdown("Ask questions about the top 1000 movies! You can ask about **plots**, **ratings**, **genres**, or **complex queries**.")

import torch
import numpy as np
from transformers import pipeline
# from datasets import load_dataset (Removed for security)
import soundfile as sf
import tempfile
from pydub import AudioSegment

# initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for API Key
# Sidebar for API Key (Disabled for Local Mode)
# with st.sidebar:
#     api_key = st.text_input("OpenAI API Key", type="password")
#     if api_key:
#         os.environ["OPENAI_API_KEY"] = api_key
#     
#     st.markdown("### Example Queries")
#     st.markdown("- Top 5 movies of 2019 by meta score")
#     st.markdown("- Movies about alien invasions")
#     st.markdown("- Top horror movies with meta score > 85")

# if not api_key:
#     st.warning("Please enter your OpenAI API Key in the sidebar to continue.")
#     st.stop()

@st.cache_resource
def load_stt_model():
    # Using OpenAI Whisper Base English (smaller, faster, less prone to hallucination on short audio)
    # Forced to CPU for debugging stability
    return pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device="cpu")

@st.cache_resource
def load_tts_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # Using Microsoft SpeechT5
    synthesiser = pipeline("text-to-speech", model="microsoft/speecht5_tts", device=device)
    # Load a default speaker embedding for SpeechT5
    try:
        # Load local secure embedding
        # embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True)
        # speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        speaker_embedding = torch.load("speaker_embedding.pt", map_location=device)
    except Exception as e:
        st.error(f"Failed to load speaker embeddings: {e}")
        speaker_embedding = None
    return synthesiser, speaker_embedding

# Sidebar


# Sidebar
# Auto-set dummy key for local mode
api_key = "ollama_local"

# Initialize Agent
if "agent" not in st.session_state:
    try:
        with st.spinner("Initializing Agent & Vector Store..."):
            st.session_state.agent = MovieAgent(api_key)
        st.success("Agent Ready!")
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        st.stop()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Audio Input
audio_value = st.audio_input("Record your question")

if audio_value:
    # Transcribe audio using Local Whisper
    try:
        with st.spinner("Transcribing (Whisper V3 Turbo)..."):
            stt_pipe = load_stt_model()
            # Convert audio_value (BytesIO) to bytes or temp file
            # Pipeline usually handles bytes or filepath. Best to save to temp file.
            # Pipeline usually handles bytes or filepath. Best to save to temp file.
            # We save as .webm first (common browser format) or just binary, then convert.
            with tempfile.NamedTemporaryFile(delete=False) as tmp_raw:
                tmp_raw.write(audio_value.read())
                tmp_raw_path = tmp_raw.name
            
            # Convert to standard WAV 16kHz using pydub
            wav_path = tmp_raw_path + ".wav"
            try:
                # Hint format as wav for the raw file if no ext, or just load
                audio = AudioSegment.from_file(tmp_raw_path) 
                
                # DEBUG: Check if audio is silent
                st.write(f"🔊 Audio Debug: Duration={len(audio)}ms, dBFS={audio.dBFS:.2f}, Channels={audio.channels}, FrameRate={audio.frame_rate}")
                if audio.dBFS < -50:
                    st.warning("⚠️ Audio seems very quiet. Check your microphone.")
                
                # Normalize audio to -20dBFS (standard speech volume)
                change_in_dBFS = -20.0 - audio.dBFS
                audio = audio.apply_gain(change_in_dBFS)

                # Ensure 16-bit 16kHz Mono
                audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                audio.export(wav_path, format="wav")
                
                # DEBUG: Play back what we are sending to the model
                st.markdown("**Debug: Processed Audio sent to Whisper:**")
                st.audio(wav_path)
                
            except Exception as e:
                st.error(f"Audio conversion failed: {e}")
                # Fallback: try using the raw file if conversion fails
                wav_path = tmp_raw_path

            # Pass the file path to pipeline (more robust than raw numpy for some versions)
            # For English-only models (.en), we must NOT pass language='english'
            result = stt_pipe(wav_path)
            user_query = result["text"]
            
            # Clean up
            try:
                os.remove(tmp_raw_path)
                if os.path.exists(wav_path):
                    os.remove(wav_path)
            except:
                pass
            
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        # Get Agent Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking (Gemma 3)..."):
                response = st.session_state.agent.run(user_query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # TTS using SpeechT5
                try:
                    with st.spinner("Generating Speech (Microsoft SpeechT5)..."):
                        tts_pipe, speaker_emb = load_tts_model()
                        if speaker_emb is not None:
                            # SpeechT5 has a 600 token limit (approx 300-400 words). 
                            # We'll truncate strictly to avoid crashes on long agent responses.
                            # A better long-term fix would be to chunk the text.
                            full_text = response
                            if len(full_text) > 400:
                                speech_text = full_text[:400] + "... (truncated for voice)"
                            else:
                                speech_text = full_text
                                
                            speech = tts_pipe(speech_text, forward_params={"speaker_embeddings": speaker_emb})
                            
                            # Validating sample rate
                            sample_rate = speech["sampling_rate"]
                            audio_data = speech["audio"]
                            
                            # Save to temp file for st.audio
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                                sf.write(tmp_out.name, audio_data.T, sample_rate)
                                output_audio_path = tmp_out.name
                                
                            st.audio(output_audio_path, autoplay=True)
                except Exception as e:
                    st.error(f"TTS Error: {e}")
                
                # Suggestions
                with st.spinner("Finding similar movies..."):
                    suggestions = st.session_state.agent.get_suggestions(response)
                    if suggestions:
                        st.markdown(suggestions)
                        st.session_state.messages.append({"role": "assistant", "content": suggestions})
                
    except Exception as e:
        st.error(f"Error processing audio: {e}")

# Text Input fallback
if prompt := st.chat_input("Or type your question here"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.agent.run(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

                # Suggestions
                with st.spinner("Finding similar movies..."):
                    suggestions = st.session_state.agent.get_suggestions(response)
                    if suggestions:
                        st.markdown(suggestions)
                        st.session_state.messages.append({"role": "assistant", "content": suggestions})
            except Exception as e:
                st.error(f"Error: {e}")
