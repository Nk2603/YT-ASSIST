
import unicodedata
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

from dotenv import load_dotenv

from app import (
    initialize_rag_chain,
    generate_study_material,
    YOUTUBE_URL as DEFAULT_URL,
)

from pdf_generator import create_document_html

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


# ============================================================
# INITIALIZATION
# ============================================================

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

    # Study material
    "generated_notes": "",
    "generated_diagrams": [],
    "notes_source_url": None,
    "generated_notes_language": "English",
    "generated_pdf": None,
    "generated_diagram_count": 0,
    "study_generation_error": None,

    # UI
    "notes_language": "English",
    "theme_mode": "Light",
}


for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = (
            value.copy() if isinstance(value, (list, dict)) else value
        )


# ============================================================
# THEME SYSTEM
# ============================================================

theme_mode = st.sidebar.selectbox(
    "🎨 Appearance",
    options=["Light", "Dark"],
    key="theme_mode",
)


THEMES = {
    "Light": {
        "bg": "#f4f3ff",
        "surface": "#ffffff",
        "surface_alt": "#eeeaff",
        "sidebar": "#e9e5ff",
        "text": "#191b50",
        "muted": "#737895",
        "border": "#ded6fa",
        "primary": "#8150e8",
        "primary_dark": "#6636ce",
        "blue": "#3998f5",
        "input": "#ffffff",
        "chat_user": "#e4eaff",
        "chat_assistant": "#f2edff",
        "shadow": "rgba(91, 77, 160, 0.08)",
        "success_bg": "#e4f8e9",
    },

    "Dark": {
        "bg": "#111426",
        "surface": "#1c2038",
        "surface_alt": "#252a48",
        "sidebar": "#191d34",
        "text": "#f0edff",
        "muted": "#b0b5d0",
        "border": "#373b60",
        "primary": "#a782ff",
        "primary_dark": "#8256df",
        "blue": "#62b0ff",
        "input": "#252a43",
        "chat_user": "#30345b",
        "chat_assistant": "#242942",
        "shadow": "rgba(0, 0, 0, 0.22)",
        "success_bg": "#193c32",
    },
}


colors = THEMES[theme_mode]

theme_variables = "\n".join(
    f"--{key.replace('_', '-')}: {value};"
    for key, value in colors.items()
)


# ============================================================
# GLOBAL THEME CSS
# ============================================================

st.markdown(
    f"""
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    :root {{
        {theme_variables}
        color-scheme: {"light" if theme_mode == "Light" else "dark"};
    }}

    .stApp {{
        background: var(--bg);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    [data-testid="stMainBlockContainer"] {{
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }}

    /* Sidebar */

    [data-testid="stSidebar"] {{
        background: var(--sidebar);
        border-right: 1px solid var(--border);
    }}

    [data-testid="stSidebar"] * {{
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }}

    /* Typography */

    h1, h2, h3, h4 {{
        color: var(--text) !important;
        font-weight: 750 !important;
    }}

    p, label, li, span {{
        color: var(--text);
    }}

    [data-testid="stCaptionContainer"] p {{
        color: var(--muted) !important;
    }}

    /* Containers */

    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 18px !important;
        box-shadow: 0 4px 16px var(--shadow);
    }}

    [data-testid="stHorizontalBlock"] {{
        gap: 1rem;
    }}

    /* Buttons */

    .stButton > button,
    .stDownloadButton > button {{
        border-radius: 13px;
        min-height: 43px;
        border: 1px solid var(--border);
        background: var(--surface);
        color: var(--text);
        font-weight: 600;
        transition: all 0.2s ease;
    }}

    .stButton > button:hover,
    .stDownloadButton > button:hover {{
        border-color: var(--primary);
        color: var(--primary);
        box-shadow: 0 5px 15px var(--shadow);
        transform: translateY(-1px);
    }}

    .stButton > button[kind="primary"] {{
        background: linear-gradient(
            100deg,
            var(--primary),
            var(--blue)
        );
        color: #ffffff !important;
        border: none;
    }}

    .stButton > button:disabled {{
        opacity: 0.55;
        cursor: not-allowed;
    }}

    /* Inputs */

    .stTextInput input,
    .stTextArea textarea,
    .stChatInput textarea,
    .stNumberInput input {{
        background: var(--input) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 13px !important;
        -webkit-text-fill-color: var(--text) !important;
    }}

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {{
        color: var(--muted) !important;
    }}

    /* Select boxes */

    [data-baseweb="select"] > div {{
        background: var(--input) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
    }}

    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"] {{
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
    }}

    [role="option"] {{
        background: var(--surface) !important;
        color: var(--text) !important;
    }}

    [role="option"]:hover {{
        background: var(--surface_alt) !important;
    }}

    /* Chat */

    [data-testid="stChatMessage"] {{
        background: var(--chat-assistant) !important;
        border: 1px solid var(--border) !important;
        border-radius: 17px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: 0 3px 12px var(--shadow);
    }}

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] strong {{
        color: var(--text) !important;
    }}

    [data-testid="stChatInput"] {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
    }}

    /* Expanders */

    [data-testid="stExpander"],
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] summary {{
        background: var(--surface) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
        border-radius: 14px;
    }}

    /* Tabs */

    [data-testid="stTabs"] [role="tab"] {{
        color: var(--muted) !important;
    }}

    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
        color: var(--primary) !important;
        border-bottom-color: var(--primary) !important;
    }}

    /* Alerts */

    [data-testid="stAlert"] {{
        border-radius: 13px;
        border-color: var(--border) !important;
    }}

    [data-testid="stAlert"] p {{
        color: var(--text) !important;
    }}

    /* Code */

    [data-testid="stCodeBlock"],
    [data-testid="stCodeBlock"] pre,
    [data-testid="stCodeBlock"] code {{
        background: var(--surface_alt) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
    }}

    /* Tables */

    [data-testid="stTable"],
    [data-testid="stTable"] table,
    [data-testid="stTable"] th,
    [data-testid="stTable"] td {{
        background: var(--surface) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
    }}

    /* Links */

    [data-testid="stMarkdownContainer"] a {{
        color: var(--blue) !important;
    }}

    /* Status pill */

    .yt-status-pill {{
        display: inline-block;
        padding: 9px 14px;
        border-radius: 999px;
        background: linear-gradient(
            110deg,
            var(--primary),
            var(--blue)
        );
        color: #ffffff !important;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-align: center;
    }}

    /* Mobile */

    @media (max-width: 768px) {{

        [data-testid="stMainBlockContainer"] {{
            padding: 0.8rem 0.8rem 5rem 0.8rem;
        }}

        h1 {{
            font-size: 1.8rem !important;
        }}

        h2 {{
            font-size: 1.4rem !important;
        }}

        h3 {{
            font-size: 1.15rem !important;
        }}

        [data-testid="stHorizontalBlock"] {{
            flex-wrap: wrap !important;
            gap: 0.8rem !important;
        }}

        .stButton > button,
        .stDownloadButton > button {{
            width: 100%;
            min-height: 46px;
        }}

        [data-testid="stChatMessage"] {{
            padding: 10px;
            border-radius: 14px;
        }}

        [data-testid="stSidebar"] {{
            max-width: 340px;
        }}

        .yt-status-pill {{
            margin: 0.25rem 0 0.75rem;
        }}

        [data-testid="stVerticalBlockBorderWrapper"] {{
            border-radius: 16px !important;
        }}
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TEXT UTILITIES
# ============================================================

def clean_notes_text(text):
    """Normalize Unicode and remove known replacement characters."""

    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)

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

    text = "".join(
        char
        for char in text
        if char in ("\n", "\t") or ord(char) >= 32
    )

    return text.strip()


def get_answer_text(response):
    """Convert common LangChain response formats to text."""

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
    """Clear all generated study material together."""

    st.session_state.generated_notes = ""
    st.session_state.generated_diagrams = []
    st.session_state.notes_source_url = None
    st.session_state.generated_notes_language = "English"
    st.session_state.generated_pdf = None
    st.session_state.generated_diagram_count = 0
    st.session_state.study_generation_error = None


def has_user_question(messages):
    """Return True if the current conversation contains a user question."""

    if not isinstance(messages, list):
        return False

    return any(
        isinstance(message, dict)
        and message.get("role") == "user"
        and bool(str(message.get("content", "")).strip())
        for message in messages
    )


def create_pdf_filename():
    """Create a safe filename for the generated PDF."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"YT_RAG_Study_Notes_{timestamp}.pdf"


# ============================================================
# CONVERSATION MANAGEMENT
# ============================================================

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

    st.session_state.chat_history = get_messages(
        conversation_id
    )

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


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎥 YT-ASSIST")

    st.caption(
        "Your personal video learning workspace"
    )

    st.button(
        "➕ New Chat",
        on_click=start_new_chat,
        use_container_width=True,
    )

    st.divider()

    st.header("💬 Chat History")

    conversations = list_conversations()

    if not conversations:
        st.caption(
            "Your saved conversations will appear here."
        )

    for conversation in conversations:

        conversation_id = conversation["id"]

        col1, col2 = st.columns([4, 1])

        with col1:

            label = (
                conversation.get("title")
                or "Untitled conversation"
            )

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
        get_conversation(current_id)
        if current_id
        else None
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

                    update_title(
                        current_id,
                        new_title.strip(),
                    )

                    st.success(
                        "Conversation title updated."
                    )

                    st.rerun()

                else:
                    st.warning("Enter a title first.")


# ============================================================
# RESTORE SAVED CONVERSATION
# ============================================================

if st.session_state.pending_chat_load:

    saved_url = st.session_state.youtube_url_input.strip()

    st.session_state.pending_chat_load = False

    if saved_url:

        try:

            with st.spinner(
                "Restoring the saved video's RAG chain..."
            ):

                restored_chain = initialize_rag_chain(
                    saved_url
                )

            st.session_state.rag_chain = restored_chain
            st.session_state.last_processed_url = saved_url

        except Exception as exc:

            st.session_state.rag_chain = None
            st.session_state.last_processed_url = None

            st.error(
                f"Could not restore the saved video: {exc}"
            )


# ============================================================
# MAIN HEADER
# ============================================================

hero_left, hero_right = st.columns(
    [3.2, 1],
    vertical_alignment="center",
)

with hero_left:

    st.markdown("# 🎥 YT-ASSIST")

    st.markdown(
        "Turn YouTube videos into an interactive learning space — "
        "ask questions, understand concepts, and create study material."
    )

with hero_right:

    st.markdown(
        """
        <div class="yt-status-pill">
            ✦ AI VIDEO LEARNING
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# YOUTUBE VIDEO INPUT
# ============================================================

with st.container(border=True):

    st.markdown("### 🔗 Add your YouTube video")

    url_col, action_col = st.columns(
        [4, 1.15],
        vertical_alignment="bottom",
    )

    with url_col:

        st.text_input(
            "YouTube Video URL",
            key="youtube_url_input",
            placeholder="Paste a YouTube video link here...",
            label_visibility="collapsed",
        )

    with action_col:

        process_button = st.button(
            "⚡ Process Video",
            type="primary",
            use_container_width=True,
        )

    active_video_url = st.session_state.last_processed_url

    current_url = st.session_state.youtube_url_input.strip()

    if active_video_url and active_video_url == current_url:

        st.success(
            "Video is ready. You can start asking questions below."
        )

        st.caption(
            f"Processed source: {active_video_url}"
        )

    elif active_video_url and active_video_url != current_url:

        st.warning(
            "The URL has changed. Process this video before continuing."
        )

    else:

        st.caption(
            "Paste a public YouTube URL and select Process Video to begin."
        )


# ============================================================
# PROCESS VIDEO
# ============================================================

if process_button:

    requested_url = st.session_state.youtube_url_input.strip()

    if not requested_url:

        st.warning(
            "Please enter a YouTube URL first."
        )

    elif (
        requested_url == st.session_state.last_processed_url
        and st.session_state.rag_chain is not None
    ):

        st.info(
            "This video is already processed."
        )

    else:

        try:

            with st.spinner(
                "Loading transcript, creating embeddings, "
                "and building the RAG chain..."
            ):

                new_chain = initialize_rag_chain(
                    requested_url
                )

            old_url = st.session_state.last_processed_url

            st.session_state.rag_chain = new_chain
            st.session_state.last_processed_url = requested_url

            if old_url != requested_url:
                clear_study_material()

            current_id = st.session_state.current_conversation_id

            if current_id:
                update_video_url(
                    current_id,
                    requested_url,
                )

            st.success(
                "Video processed successfully."
            )

        except Exception as exc:

            st.error(
                f"Video processing failed: {exc}"
            )


# ============================================================
# CHAT INTERFACE
# ============================================================

with st.container(border=True):

    st.subheader("💬 Video Chat")

    st.caption(
        "Ask anything from the processed video transcript. "
        "Answers are grounded in the video content."
    )

    current_id = st.session_state.current_conversation_id

    if current_id:

        st.session_state.chat_history = get_messages(
            current_id
        )

    else:

        st.session_state.chat_history = []

    # Display previous messages

    for message in st.session_state.chat_history:

        role = message.get("role", "assistant")

        if role not in ("user", "assistant"):
            role = "assistant"

        with st.chat_message(role):

            st.markdown(
                message.get("content", "")
            )

    prompt = st.chat_input(
        "Ask a question about the video..."
    )

    if prompt:

        chain = st.session_state.rag_chain

        processed_url = st.session_state.last_processed_url

        current_url = st.session_state.youtube_url_input.strip()

        if chain is None or processed_url != current_url:

            st.warning(
                "Please process the current video before asking questions."
            )

        else:

            current_id = st.session_state.current_conversation_id

            if current_id is None:

                current_id = create_conversation(
                    video_url=current_url,
                    title=prompt[:45],
                )

                st.session_state.current_conversation_id = current_id

            add_message(
                current_id,
                "user",
                prompt,
            )

            with st.chat_message("user"):
                st.markdown(prompt)

            try:

                with st.chat_message("assistant"):

                    with st.spinner(
                        "Searching the transcript..."
                    ):

                        raw_response = chain.invoke(prompt)

                    response = get_answer_text(
                        raw_response
                    )

                    if not response:
                        response = (
                            "The model returned an empty answer."
                        )

                    st.markdown(response)

                add_message(
                    current_id,
                    "assistant",
                    response,
                )

            except Exception as exc:

                st.error(
                    f"Answer generation failed: {exc}"
                )

                add_message(
                    current_id,
                    "assistant",
                    f"Could not generate an answer: {exc}",
                )

            st.rerun()


# ============================================================
# REFRESH CURRENT CHAT HISTORY
# ============================================================

current_id = st.session_state.current_conversation_id

if current_id:
    st.session_state.chat_history = get_messages(current_id)


# ============================================================
# LEARNING RESOURCES
# ============================================================

st.markdown("### ✨ Learning Resources")

resource_cols = st.columns(3, gap="medium")


# ------------------------------------------------------------
# STUDY NOTES
# ------------------------------------------------------------

with resource_cols[0]:

    with st.container(border=True):

        st.markdown("#### 📝 Study Notes")

        st.caption("🟢 Available")

        st.write(
            "Generate complete video notes with explanations, "
            "examples, formulas and relevant educational visuals."
        )

        notes_language = st.selectbox(
            "Notes language",
            options=["English", "Hindi", "Hinglish"],
            key="notes_language",
        )

        question_exists = has_user_question(
            st.session_state.chat_history
        )

        video_ready = (
            st.session_state.rag_chain is not None
            and st.session_state.last_processed_url
            == st.session_state.youtube_url_input.strip()
        )

        generate_notes_button = st.button(
            "✨ Generate Study Notes",
            type="primary",
            use_container_width=True,
            key="generate_notes_from_resource",
            disabled=not (video_ready and question_exists),
        )

        if not video_ready:

            st.caption(
                "Process a YouTube video to enable study notes."
            )

        elif not question_exists:

            st.caption(
                "Ask at least one question about the video before generating notes."
            )


# ------------------------------------------------------------
# PRESENTATION
# ------------------------------------------------------------

with resource_cols[1]:

    with st.container(border=True):

        st.markdown("#### 📊 Presentation")

        st.caption("🟠 Coming Soon")

        st.write(
            "Create an organized presentation from "
            "your YouTube video content."
        )


# ------------------------------------------------------------
# QUIZ PRACTICE
# ------------------------------------------------------------

with resource_cols[2]:

    with st.container(border=True):

        st.markdown("#### 🧠 Quiz Practice")

        st.caption("🟠 Coming Soon")

        st.write(
            "Test your understanding with "
            "video-based practice questions."
        )


# ============================================================
# STUDY NOTES GENERATION
# ============================================================

if generate_notes_button:

    active_video_url = st.session_state.last_processed_url

    current_url = st.session_state.youtube_url_input.strip()

    if not active_video_url or st.session_state.rag_chain is None:

        st.warning(
            "Please process a YouTube video first."
        )

    elif active_video_url != current_url:

        st.warning(
            "The current URL differs from the processed video. "
            "Please process the current video first."
        )

    elif not has_user_question(st.session_state.chat_history):

        st.warning(
            "Ask at least one question about the video before generating notes."
        )

    else:

        clear_study_material()

        try:

            with st.spinner(
                "Creating your complete study notes, "
                "analyzing video visuals and generating PDF..."
            ):

                result = generate_study_material(
                    youtube_url=active_video_url,
                    chat_history=st.session_state.chat_history,
                    language=notes_language,
                )

            if not isinstance(result, tuple) or len(result) != 3:

                raise ValueError(
                    "generate_study_material() must return "
                    "(notes, diagram_images, pdf_bytes)."
                )

            notes, diagram_images, pdf_bytes = result

            notes = clean_notes_text(notes)

            if not notes:
                raise ValueError(
                    "Generated notes are empty."
                )

            # Diagrams are optional.
            # Notes-only PDF is also valid.

            if not isinstance(diagram_images, list):
                diagram_images = []

            if (
                not isinstance(pdf_bytes, bytes)
                or not pdf_bytes.startswith(b"%PDF")
            ):

                raise ValueError(
                    "Generated PDF is invalid."
                )

            # Save generated material in session state.
            # These operations do not write to chat history.

            st.session_state.generated_notes = notes

            st.session_state.generated_diagrams = diagram_images

            st.session_state.notes_source_url = active_video_url

            st.session_state.generated_notes_language = notes_language

            st.session_state.generated_pdf = pdf_bytes

            st.session_state.generated_diagram_count = len(
                diagram_images
            )

            st.session_state.study_generation_error = None

            st.success(
                "Your study notes are ready!"
            )

            st.rerun()

        except Exception as exc:

            st.session_state.study_generation_error = str(exc)

            st.error(
                f"Study notes generation failed: {exc}"
            )


# ============================================================
# STUDY NOTES WORKSPACE
# ============================================================

if st.session_state.generated_notes:

    if (
        st.session_state.notes_source_url
        != st.session_state.last_processed_url
    ):

        st.warning(
            "These notes belong to another video. "
            "Generate notes again for the current video."
        )

    else:

        st.divider()

        st.markdown(
            "# 📝 Your Study Notes"
        )

        diagram_count = st.session_state.generated_diagram_count

        if diagram_count:

            visual_status = (
                f"{diagram_count} educational visual(s) included"
            )

        else:

            visual_status = (
                "No suitable video visuals were detected; "
                "the PDF contains notes without fabricated diagrams."
            )

        st.caption(
            f"Language: {st.session_state.generated_notes_language} "
            f"| {visual_status}"
        )

        # ----------------------------------------------------
        # NOTEBOOK STYLE PREVIEW
        # ----------------------------------------------------

        with st.container(border=True):

            st.markdown(
                "### 📖 Handwritten Notes Preview"
            )

            try:

                preview_html = create_document_html(
                    notes=st.session_state.generated_notes,
                    title="YouTube Video Study Notes",
                    language=st.session_state.generated_notes_language,
                    diagram_images=st.session_state.generated_diagrams,
                )

                st.iframe(preview_html, height=1100, scrolling=True)
            except Exception as exc:

                st.error(
                    f"Could not render handwritten notes preview: {exc}"
                )

                st.markdown(
                    st.session_state.generated_notes
                )

        # ----------------------------------------------------
        # PDF DOWNLOAD
        # ----------------------------------------------------

        pdf_bytes = st.session_state.generated_pdf

        if pdf_bytes:

            with st.container(border=True):

                st.markdown(
                    "### 📥 Download Your Study Material"
                )

                st.caption(
                    "Download your complete video notes as a PDF. "
                    "Relevant visuals are included when available."
                )

                st.download_button(
                    label="📥 Download Complete Study Notes PDF",
                    data=pdf_bytes,
                    file_name=create_pdf_filename(),
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                    key="download_complete_study_pdf",
                )

        # ----------------------------------------------------
        # REGENERATE NOTES
        # ----------------------------------------------------

        col_regenerate, col_clear = st.columns(2)

        with col_regenerate:

            if st.button(
                "🔄 Regenerate Study Notes",
                use_container_width=True,
                key="regenerate_study_notes",
            ):

                clear_study_material()
                st.rerun()

        # ----------------------------------------------------
        # CLEAR NOTES
        # ----------------------------------------------------

        with col_clear:

            if st.button(
                "🧹 Clear Study Notes",
                use_container_width=True,
                key="clear_study_material_button",
            ):

                clear_study_material()
                st.rerun()


# ============================================================
# GENERATION ERROR
# ============================================================

if st.session_state.study_generation_error:

    with st.expander("⚠️ Study notes generation details"):

        st.error(
            st.session_state.study_generation_error
        )
