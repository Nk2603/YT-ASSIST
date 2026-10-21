
import unicodedata

import streamlit as st
from dotenv import load_dotenv

from app import (
    initialize_rag_chain,
    generate_study_material,
    YOUTUBE_URL as DEFAULT_URL,
)

from services.chat_history_service import (
    init_db,
    create_conversation,
    list_conversations,
    get_conversation,
    get_messages,
    add_message,
    update_title,
    update_video_url,
    delete_conversation,
)


# ==================================================
# INITIALIZATION
# ==================================================

load_dotenv()

st.set_page_config(
    page_title="YT-RAG-Assist",
    page_icon="🎥",
    layout="wide",
)

init_db()

DEFAULTS = {
    "rag_chain": None,
    "last_processed_url": None,
    "current_conversation_id": None,
    "chat_history": [],
    "youtube_url_input": DEFAULT_URL,
    "pending_chat_load": False,
    "generated_notes": "",
    "notes_source_url": None,
    "generated_notes_language": "English",
    "generated_pdf": None,
    "generated_diagram_count": 0,
    "study_generation_error": None,
    "notes_language": "English",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = (
            value.copy() if isinstance(value, (list, dict)) else value
        )


# ==================================================
# TEXT UTILITIES
# ==================================================

def clean_notes_text(text):
    """
    Normalize Unicode text and remove known placeholder squares.

    This helps with literal placeholder characters in the source text.
    Missing PDF glyphs must be fixed through font rendering, not text
    cleanup alone.
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)

    # Remove known square/replacement placeholders without inserting
    # spaces inside words.
    unwanted = (
        "\u25a0",
        "\u25aa",
        "\u25ab",
        "\u25fb",
        "\u25fc",
        "\ufffd",
    )

    for character in unwanted:
        text = text.replace(character, "")

    # Remove control characters except tabs and newlines.
    text = "".join(
        char
        for char in text
        if char in ("\n", "\t") or ord(char) >= 32
    )

    return text.strip()


def get_answer_text(response):
    """Convert common LangChain response formats to a string."""
    if isinstance(response, dict):
        response = (
            response.get("answer")
            or response.get("result")
            or response.get("output")
            or str(response)
        )

    if not isinstance(response, str):
        response = str(response)

    return response.strip()


def clear_study_material():
    """Clear generated notes, diagrams metadata, and PDF together."""
    st.session_state.generated_notes = ""
    st.session_state.notes_source_url = None
    st.session_state.generated_notes_language = "English"
    st.session_state.generated_pdf = None
    st.session_state.generated_diagram_count = 0
    st.session_state.study_generation_error = None


# ==================================================
# CONVERSATION MANAGEMENT
# ==================================================

def start_new_chat():
    st.session_state.current_conversation_id = None
    st.session_state.chat_history = []
    st.session_state.rag_chain = None
    st.session_state.last_processed_url = None
    st.session_state.pending_chat_load = False

    clear_study_material()


def select_conversation(conversation_id):
    conversation = get_conversation(conversation_id)

    if not conversation:
        st.error("The selected conversation could not be found.")
        return

    st.session_state.current_conversation_id = conversation_id
    st.session_state.chat_history = get_messages(conversation_id)

    st.session_state.youtube_url_input = (
        conversation.get("video_url") or DEFAULT_URL
    )

    st.session_state.rag_chain = None
    st.session_state.last_processed_url = None
    st.session_state.pending_chat_load = True

    clear_study_material()


def remove_conversation(conversation_id):
    delete_conversation(conversation_id)

    if st.session_state.current_conversation_id == conversation_id:
        start_new_chat()


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.header("🎥 Video Settings")

    st.button(
        "➕ New Chat",
        on_click=start_new_chat,
        use_container_width=True,
    )

    st.text_input(
        "YouTube Video URL",
        key="youtube_url_input",
        placeholder="Paste a YouTube video URL",
    )

    process_button = st.button(
        "⚙️ Process Video",
        type="primary",
        use_container_width=True,
    )

    st.divider()
    st.header("💬 Chat History")

    conversations = list_conversations()

    if not conversations:
        st.caption("Your saved conversations will appear here.")

    for conversation in conversations:
        conversation_id = conversation["id"]
        col1, col2 = st.columns([4, 1])

        with col1:
            label = conversation.get("title") or "Untitled conversation"

            if conversation_id == st.session_state.current_conversation_id:
                label = "📌 " + label

            st.button(
                label,
                key=f"open_chat_{conversation_id}",
                on_click=select_conversation,
                args=(conversation_id,),
                use_container_width=True,
            )

        with col2:
            st.button(
                "🗑️",
                key=f"delete_chat_{conversation_id}",
                help="Delete this conversation",
                on_click=remove_conversation,
                args=(conversation_id,),
                use_container_width=True,
            )

    current_id = st.session_state.current_conversation_id
    current_conversation = (
        get_conversation(current_id) if current_id else None
    )

    if current_conversation:
        with st.expander("✏️ Rename Current Chat"):
            rename_key = f"rename_title_{current_id}"

            if rename_key not in st.session_state:
                st.session_state[rename_key] = (
                    current_conversation.get("title") or ""
                )

            new_title = st.text_input(
                "Conversation title",
                key=rename_key,
            )

            if st.button(
                "Save Title",
                key=f"save_title_{current_id}",
            ):
                if new_title.strip():
                    update_title(current_id, new_title.strip())
                    st.success("Conversation title updated.")
                    st.rerun()
                else:
                    st.warning("Enter a title first.")


# ==================================================
# RESTORE A SAVED CONVERSATION
# ==================================================

if st.session_state.pending_chat_load:
    saved_url = st.session_state.youtube_url_input.strip()
    st.session_state.pending_chat_load = False

    if saved_url:
        try:
            with st.spinner("Restoring the saved video's RAG chain..."):
                restored_chain = initialize_rag_chain(saved_url)

            st.session_state.rag_chain = restored_chain
            st.session_state.last_processed_url = saved_url

        except Exception as exc:
            st.session_state.rag_chain = None
            st.session_state.last_processed_url = None

            st.error(f"Could not restore the saved video: {exc}")


# ==================================================
# PROCESS VIDEO
# ==================================================

if process_button:
    requested_url = st.session_state.youtube_url_input.strip()

    if not requested_url:
        st.warning("Please enter a YouTube URL first.")

    elif (
        requested_url == st.session_state.last_processed_url
        and st.session_state.rag_chain is not None
    ):
        st.info("This video is already processed.")

    else:
        try:
            with st.spinner(
                "Loading transcript, creating embeddings, and building "
                "the RAG chain..."
            ):
                new_chain = initialize_rag_chain(requested_url)

            old_url = st.session_state.last_processed_url

            st.session_state.rag_chain = new_chain
            st.session_state.last_processed_url = requested_url

            if old_url != requested_url:
                clear_study_material()

            current_id = st.session_state.current_conversation_id

            if current_id:
                update_video_url(current_id, requested_url)

            st.success("Video processed successfully.")

        except Exception as exc:
            st.error(f"Video processing failed: {exc}")


# ==================================================
# MAIN HEADER
# ==================================================

st.title("🎥 YT-RAG-Assist")

st.markdown(
    """
    Ask questions about a YouTube video and generate a complete study
    document with explanations, examples, revision points, and visual diagrams.

    **Technology:** LangChain · FAISS · Gemini Embeddings · Groq
    """
)

active_video_url = st.session_state.last_processed_url
current_url = st.session_state.youtube_url_input.strip()

if active_video_url:
    st.caption(f"Processed video: {active_video_url}")

    if active_video_url != current_url:
        st.warning(
            "The URL has changed. Select **Process Video** before "
            "asking questions or generating study material."
        )
else:
    st.info(
        "Paste a YouTube URL in the sidebar and select **Process Video**."
    )

st.divider()


# ==================================================
# CHAT INTERFACE
# ==================================================

st.subheader("💬 Ask Questions About the Video")

current_id = st.session_state.current_conversation_id

if current_id:
    st.session_state.chat_history = get_messages(current_id)
else:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    role = message.get("role", "assistant")

    if role not in ("user", "assistant"):
        role = "assistant"

    with st.chat_message(role):
        st.markdown(message.get("content", ""))


prompt = st.chat_input("Ask a question about the video...")

if prompt:
    chain = st.session_state.rag_chain
    processed_url = st.session_state.last_processed_url
    current_url = st.session_state.youtube_url_input.strip()

    if chain is None or processed_url != current_url:
        st.warning("Please process the current video before asking questions.")

    else:
        current_id = st.session_state.current_conversation_id

        if current_id is None:
            current_id = create_conversation(
                video_url=current_url,
                title=prompt[:45],
            )

            st.session_state.current_conversation_id = current_id

        add_message(current_id, "user", prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Searching the transcript..."):
                    raw_response = chain.invoke(prompt)

                response = get_answer_text(raw_response)

                if not response:
                    response = "The model returned an empty answer."

                st.markdown(response)

            add_message(current_id, "assistant", response)

        except Exception as exc:
            st.error(f"Answer generation failed: {exc}")

            add_message(
                current_id,
                "assistant",
                f"Could not generate an answer: {exc}",
            )

        st.rerun()


# ==================================================
# COMPLETE STUDY NOTES + VISUAL DIAGRAMS
# ==================================================

st.divider()
st.subheader("📚 Complete Video Study Notes")

st.markdown(
    """
    Generate one PDF containing:

    - Complete topic-wise study notes
    - Important definitions and examples
    - Tables, key points, and quick revision
    - Visual diagrams and flowcharts embedded as images

    Diagram source code will not be displayed as the final output.
    """
)

notes_language = st.selectbox(
    "Choose study notes language",
    options=["English", "Hindi", "Hinglish"],
    key="notes_language",
)

if st.button(
    "✨ Generate Notes + Diagrams PDF",
    type="primary",
    use_container_width=True,
    key="generate_complete_study_pdf",
):
    active_video_url = st.session_state.last_processed_url
    current_url = st.session_state.youtube_url_input.strip()

    if not active_video_url or st.session_state.rag_chain is None:
        st.warning("Please process a video first.")

    elif active_video_url != current_url:
        st.warning(
            "The current URL differs from the processed video. "
            "Process the video before generating study material."
        )

    else:
        # Prevent an old PDF from being offered if generation fails.
        clear_study_material()

        try:
            with st.spinner(
                "Generating notes, rendering diagrams, and preparing the PDF..."
            ):
                notes, diagram_images, pdf_bytes = generate_study_material(
                    youtube_url=active_video_url,
                    chat_history=st.session_state.chat_history,
                    language=notes_language,
                )

            notes = clean_notes_text(notes)

            if not notes:
                raise ValueError("The generated notes are empty.")

            if not isinstance(diagram_images, list) or not diagram_images:
                raise ValueError(
                    "No rendered diagram images were returned. "
                    "Check the diagram-rendering setup in app.py."
                )

            if not isinstance(pdf_bytes, bytes) or not pdf_bytes.startswith(
                b"%PDF"
            ):
                raise ValueError("The generated PDF is invalid.")

            st.session_state.generated_notes = notes
            st.session_state.notes_source_url = active_video_url
            st.session_state.generated_notes_language = notes_language
            st.session_state.generated_pdf = pdf_bytes
            st.session_state.generated_diagram_count = len(diagram_images)
            st.session_state.study_generation_error = None

            st.success(
                "Your complete study document is ready with "
                f"{len(diagram_images)} visual diagrams."
            )

        except Exception as exc:
            st.session_state.study_generation_error = str(exc)
            st.error(f"Study material generation failed: {exc}")

            st.caption(
                "Check the Groq/Gemini API keys, transcript availability, "
                "font files, and Mermaid CLI installation if the error persists."
            )


# ==================================================
# NOTES PREVIEW + SINGLE PDF DOWNLOAD
# ==================================================

if st.session_state.study_generation_error:
    st.caption(
        f"Last generation error: "
        f"{st.session_state.study_generation_error}"
    )

if st.session_state.generated_notes:
    if (
        st.session_state.notes_source_url
        != st.session_state.last_processed_url
    ):
        st.info(
            "These notes belong to another video. "
            "Generate the study material again for the current video."
        )

    else:
        st.markdown("### 📝 Study Notes Preview")

        st.markdown(st.session_state.generated_notes)

        st.caption(
            "The downloaded PDF also contains the rendered visual diagrams."
        )

        pdf_bytes = st.session_state.generated_pdf

        if pdf_bytes:
            st.download_button(
                label="📥 Download Complete Study Notes PDF",
                data=pdf_bytes,
                file_name="youtube_study_notes.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download_complete_study_pdf",
            )

        if st.button(
            "🧹 Clear Generated Study Material",
            key="clear_study_material_button",
        ):
            clear_study_material()
            st.rerun()
