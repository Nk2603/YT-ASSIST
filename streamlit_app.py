import streamlit as st
import os
from dotenv import load_dotenv
from app import initialize_rag_chain, YOUTUBE_URL as DEFAULT_URL

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="🎥",
    layout="centered"
)

st.title("🎥 YouTube RAG Assistant")
st.markdown("""
    Ask questions about a YouTube video's content. 
    This app uses LangChain, FAISS, and Gemini/Groq to provide answers based on the video transcript.
""")

# --- Sidebar / Configuration ---
with st.sidebar:
    st.header("Settings")
    youtube_url = st.text_input("YouTube Video URL", value=DEFAULT_URL)
    process_button = st.button("Process Video")

# --- Session State Management ---
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_processed_url" not in st.session_state:
    st.session_state.last_processed_url = None

# --- RAG Chain Initialization & URL Change Logic ---
if process_button or (youtube_url != st.session_state.last_processed_url):
    with st.spinner("Processing video and initializing RAG chain..."):
        try:
            # Re-initialize the chain with the NEW URL
            st.session_state.rag_chain = initialize_rag_chain(youtube_url)
            
            # Reset history and update last processed URL
            st.session_state.chat_history = []
            st.session_state.last_processed_url = youtube_url
            
            st.success(f"Video processed successfully: {youtube_url}")
            # st.rerun() # Optional: rerun to clear previous messages immediately if needed
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")

# --- Chat Interface ---
st.divider()

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about the video..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    if st.session_state.rag_chain:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke(prompt)
                    message_placeholder.markdown(response)
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
    else:
        st.warning("Please process a video first or check your API keys.")
