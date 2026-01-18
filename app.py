import streamlit as st
import os
from agent import MovieAgent
# import langchain
# langchain.debug = True

st.set_page_config(page_title="IMDB Voice Agent", page_icon="🎬")

st.title("🎬 IMDB Movie Voice Agent")
st.markdown("Ask questions about the top 1000 movies! You can ask about **plots**, **ratings**, **genres**, or **complex queries**.")

# initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for API Key
with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    st.markdown("### Example Queries")
    st.markdown("- Top 5 movies of 2019 by meta score")
    st.markdown("- Movies about alien invasions")
    st.markdown("- Top horror movies with meta score > 85")

if not api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar to continue.")
    st.stop()

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
    # Transcribe audio using OpenAI
    try:
        client = st.session_state.agent.llm.client # Access underlying client or create new
        # Actually easier to just use openai directly
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        with st.spinner("Transcribing..."):
            transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_value
            )
            user_query = transcription.text
            
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        # Get Agent Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.run(user_query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # TTS
                response_audio = client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=response
                )
                response_audio.stream_to_file("output.mp3")
                st.audio("output.mp3", autoplay=True)
                
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
            except Exception as e:
                st.error(f"Error: {e}")
