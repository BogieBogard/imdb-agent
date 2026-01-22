import streamlit as st
import os
import warnings
from agent import MovieAgent
# import langchain
# langchain.debug = True

# Suppress known warnings from third-party libraries (Kokoro TTS, PyTorch, Transformers)
# These are internal library warnings that don't affect functionality
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.filterwarnings("ignore", category=UserWarning, module="kokoro.istftnet")
# Suppress transformers deprecation warnings that we can't fix (internal to transformers library)
# return_token_timestamps warning comes from WhisperFeatureExtractor initialization - will be fixed in transformers v5
# forced_decoder_ids warning appears for English-only models (whisper-base.en) which don't support
# the new language/task API - will be fixed in transformers v5
warnings.filterwarnings("ignore", message=".*return_token_timestamps.*")
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")

st.set_page_config(page_title="IMDB Voice Agent", page_icon="🎬")

st.title("🎬 IMDB Movie Voice Agent")
st.markdown("Ask questions about the top 1000 movies! You can ask about **plots**, **ratings**, **genres**, or **complex queries**.")

import torch
import numpy as np
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
    # Import here to avoid Streamlit caching issues
    from transformers import pipeline
    # Using OpenAI Whisper Base English (smaller, faster, less prone to hallucination on short audio)
    # Forced to CPU for debugging stability
    # Create pipeline - warnings about return_token_timestamps and forced_decoder_ids are from
    # internal transformers code and will be fixed in transformers v5
    # Note: whisper-base.en is English-only, so we cannot use language/task parameters
    # (those are only for multilingual models)
    return pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device="cpu")

@st.cache_resource
def load_tts_model():
    # Upgrade to Kokoro-82M (High quality, local, fast)
    from kokoro import KPipeline
    import torch
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # 'a' = American English
    pipeline = KPipeline(lang_code='a', device=device)
    
    # We return the pipeline and a default voice style
    # 'am_michael' is a good generic American Male voice included in Kokoro
    # 'af_sarah' is a good generic American Female voice
    # 'af_heart' is the highest rated voice (Grade A) in VOICES.md
    return pipeline, "af_heart"

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
            # Note: whisper-base.en is English-only, so we cannot pass language/task parameters
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
                
                # TTS using Kokoro-82M
                try:
                    with st.spinner("Generating Speech (Kokoro-82M)..."):
                        tts_pipeline, voice_style = load_tts_model()
                        
                        # Kokoro is a generator, it yields audio chunks.
                        # We need to collect them to play the full audio.
                        # Using the generator allows for streaming if we wanted, but for now we'll stitch.
                        generator = tts_pipeline(
                            response, 
                            voice=voice_style,
                            speed=1, 
                            split_pattern=r'\n+'
                        )
                        
                        all_audio = []
                        for _, _, audio_chunk in generator:
                            all_audio.append(audio_chunk)
                            
                        if all_audio:
                            import numpy as np
                            # Concatenate all chunks
                            complete_audio = np.concatenate(all_audio)
                            
                            # Validating sample rate (Kokoro defaults to 24000)
                            sample_rate = 24000
                            
                            # Save to temp file for st.audio
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                                sf.write(tmp_out.name, complete_audio, sample_rate)
                                output_audio_path = tmp_out.name
                                
                            st.audio(output_audio_path, autoplay=True)
                        else:
                            st.warning("TTS generated no audio.")
                            
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
