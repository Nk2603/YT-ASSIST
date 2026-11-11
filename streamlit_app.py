import re
import time
import unicodedata
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from app import (
    initialize_rag_chain,
    generate_study_material,
    generate_video_quiz,
    normalize_youtube_url,
    YOUTUBE_URL as DEFAULT_URL,
)
from pdf_generator import create_document_html
from quiz_generation import evaluate_quiz_attempt

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
    # Chat / video
    "rag_chain": None,
    "last_processed_url": None,
    "current_conversation_id": None,
    "chat_history": [],
    "youtube_url_input": DEFAULT_URL,
    "pending_chat_load": False,

    # Notes
    "generated_notes": "",
    "generated_diagrams": [],
    "notes_source_url": None,
    "generated_notes_language": "English",
    "generated_pdf": None,
    "generated_diagram_count": 0,
    "study_generation_error": None,

    # Appearance / UI
    "notes_language": "English",
    "theme_mode": "Light",
    "active_page": "home",

       # Quiz
    "quiz_data": None,
    "quiz_result": None,
    "quiz_source_url": None,
    "quiz_started_at": None,
    "quiz_attempt_history": [],
    "quiz_attempt_id": None,
    "quiz_attempt_recorded_id": None,
    "quiz_question_index": 0,

    # Persistent quiz answers and question history
    "quiz_saved_answers": {},
    "quiz_question_history_by_video": {},

    # Resource origin snapshot
    "resource_origin_conversation_id": None,
    "resource_origin_chat_history": [],
    "resource_origin_youtube_url": None,
    "resource_origin_last_processed_url": None,
    "resource_origin_rag_chain": None,
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
# GLOBAL CSS
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
# TEXT / DATA UTILITIES
# ============================================================

def clean_notes_text(value):
    """Normalize Unicode and remove replacement/square characters."""
    if not isinstance(value, str):
        return ""

    value = unicodedata.normalize("NFC", value)

    unwanted = (
        "\u25a0",
        "\u25aa",
        "\u25ab",
        "\u25fb",
        "\u25fc",
        "\ufffd",
    )

    for character in unwanted:
        value = value.replace(character, "")

    value = "".join(
        character
        for character in value
        if character in ("\n", "\t") or ord(character) >= 32
    )

    return value.strip()


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
    """Clear generated notes and PDF without changing the current chat."""
    st.session_state.generated_notes = ""
    st.session_state.generated_diagrams = []
    st.session_state.notes_source_url = None
    st.session_state.generated_notes_language = "English"
    st.session_state.generated_pdf = None
    st.session_state.generated_diagram_count = 0
    st.session_state.study_generation_error = None


def clear_quiz_answers():
    """Clear old widget states and saved answers."""
    for key in list(st.session_state.keys()):
        if key.startswith("quiz_answer_") or key.startswith("quiz_multi_"):
            del st.session_state[key]

    st.session_state.quiz_saved_answers = {}


def reset_current_quiz():
    """Reset the active quiz without deleting attempt history."""
    clear_quiz_answers()

    st.session_state.quiz_data = None
    st.session_state.quiz_result = None
    st.session_state.quiz_source_url = None
    st.session_state.quiz_started_at = None
    st.session_state.quiz_attempt_id = None
    st.session_state.quiz_attempt_recorded_id = None
    st.session_state.quiz_question_index = 0
    
def save_quiz_answer(question_id, widget_key):
    """
    Save a radio or text-input answer before Streamlit
    removes the widget from the active page.
    """
    answers = dict(
        st.session_state.get("quiz_saved_answers", {})
    )

    answers[str(question_id)] = st.session_state.get(widget_key)

    st.session_state.quiz_saved_answers = answers


def save_quiz_multi_answer(question_id, option_id, widget_key):
    """Save multiple-select checkbox changes."""
    answers = dict(
        st.session_state.get("quiz_saved_answers", {})
    )

    question_id = str(question_id)
    option_id = str(option_id)

    selected = set(
        answers.get(question_id, [])
    )

    if st.session_state.get(widget_key, False):
        selected.add(option_id)
    else:
        selected.discard(option_id)

    answers[question_id] = sorted(selected)

    st.session_state.quiz_saved_answers = answers    

def has_user_question(messages):
    """Check whether the conversation contains at least one user message."""
    if not isinstance(messages, list):
        return False

    return any(
        isinstance(message, dict)
        and message.get("role") == "user"
        and bool(str(message.get("content", "")).strip())
        for message in messages
    )


def create_pdf_filename():
    """Generate a unique PDF filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"YT_RAG_Study_Notes_{timestamp}.pdf"


def validate_quiz_before_display(quiz):
    """
    Perform basic UI-side checks.

    The quiz generator should perform the detailed question validation.
    This check prevents empty, duplicate or obviously incomplete questions
    from being displayed to the learner.
    """
    if not isinstance(quiz, dict):
        raise ValueError("The quiz response is not a valid dictionary.")

    questions = quiz.get("questions")

    if not isinstance(questions, list) or not questions:
        raise ValueError("The quiz does not contain any questions.")

    seen_ids = set()
    seen_questions = set()

    vague_phrases = (
        "what is this",
        "what does this mean",
        "which of the following is correct about this",
        "what happens in the above example",
        "based on the above",
        "as mentioned above",
        "in this example",
        "this concept",
        "the above concept",
    )

    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            raise ValueError(f"Question {index} has an invalid format.")

        question_text = str(question.get("question", "")).strip()

        if not question_text:
            raise ValueError(f"Question {index} has no question text.")

        words = re.findall(r"\b[\w'-]+\b", question_text)

        if len(words) < 7:
            raise ValueError(
                f"Question {index} is too short or incomplete: "
                f"{question_text}"
            )

        lowered = question_text.lower()

        if any(phrase in lowered for phrase in vague_phrases):
            raise ValueError(
                f"Question {index} contains an unclear reference: "
                f"{question_text}"
            )

        normalized_question = re.sub(
            r"\s+",
            " ",
            question_text.lower(),
        )

        if normalized_question in seen_questions:
            raise ValueError(
                f"Duplicate question detected at question {index}."
            )

        seen_questions.add(normalized_question)

        question_id = str(question.get("id", "")).strip()

        if not question_id:
            raise ValueError(
                f"Question {index} is missing its unique ID."
            )

        if question_id in seen_ids:
            raise ValueError(
                f"Duplicate question ID detected: {question_id}"
            )

        seen_ids.add(question_id)

        question_type = str(question.get("type", "mcq")).lower()

        if question_type in (
            "mcq",
            "multiple_select",
            "scenario",
        ):
            options = question.get("options", [])

            if not isinstance(options, list) or len(options) < 2:
                raise ValueError(
                    f"Question {index} does not have enough answer options."
                )

            option_ids = [
                str(option.get("id", "")).strip()
                for option in options
                if isinstance(option, dict)
            ]

            if len(option_ids) != len(options) or any(
                not option_id for option_id in option_ids
            ):
                raise ValueError(
                    f"Question {index} has an invalid option format."
                )

            if len(set(option_ids)) != len(option_ids):
                raise ValueError(
                    f"Question {index} contains duplicate option IDs."
                )

    return questions


def make_attempt_id():
    """Create a unique ID for a quiz attempt."""
    return f"{time.time():.6f}"

# ============================================================
# CONVERSATION MANAGEMENT
# ============================================================

def start_new_chat():
    """Start a clean conversation."""
    st.session_state.current_conversation_id = None
    st.session_state.chat_history = []

    st.session_state.rag_chain = None
    st.session_state.last_processed_url = None
    st.session_state.youtube_url_input = DEFAULT_URL

    st.session_state.pending_chat_load = False
    st.session_state.active_page = "home"

    clear_study_material()
    reset_current_quiz()


def select_conversation(conversation_id):
    """Load a saved conversation and restore its video context."""
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
    st.session_state.active_page = "home"

    clear_study_material()
    reset_current_quiz()


def remove_conversation(conversation_id):
    """Delete a saved conversation."""
    delete_conversation(conversation_id)

    if st.session_state.current_conversation_id == conversation_id:
        start_new_chat()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🎥 YT-ASSIST")
    st.caption("Your personal video learning workspace")

    st.button(
        "➕ New Chat",
        on_click=start_new_chat,
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
                    update_title(current_id, new_title.strip())
                    st.success("Conversation title updated.")
                    st.rerun()
                else:
                    st.warning("Enter a title first.")


# ============================================================
# RESTORE SAVED CONVERSATION
# ============================================================

if st.session_state.pending_chat_load:
    saved_url = (
        st.session_state.youtube_url_input or ""
    ).strip()

    st.session_state.pending_chat_load = False

    if saved_url:
        try:
            with st.spinner(
                "Restoring the saved video's RAG chain..."
            ):
                restored_chain = initialize_rag_chain(saved_url)

            st.session_state.rag_chain = restored_chain
            st.session_state.last_processed_url = saved_url

        except Exception as exc:
            st.session_state.rag_chain = None
            st.session_state.last_processed_url = None

            st.error(
                f"Could not restore the saved video: {exc}"
            )


# ============================================================
# VIDEO / CHAT HELPERS
# ============================================================

def video_is_ready():
    current_url = (
        st.session_state.youtube_url_input or ""
    ).strip()

    return (
        st.session_state.rag_chain is not None
        and bool(current_url)
        and st.session_state.last_processed_url == current_url
    )


def capture_resource_origin():
    """
    Save the exact chat and video context before opening Notes or Quiz.

    This snapshot is important for unsaved chats too, where the conversation
    ID may still be None.
    """
    st.session_state.resource_origin_conversation_id = (
        st.session_state.current_conversation_id
    )

    st.session_state.resource_origin_chat_history = list(
        st.session_state.chat_history or []
    )

    st.session_state.resource_origin_youtube_url = (
        st.session_state.youtube_url_input
    )

    st.session_state.resource_origin_last_processed_url = (
        st.session_state.last_processed_url
    )

    st.session_state.resource_origin_rag_chain = (
        st.session_state.rag_chain
    )


def return_to_originating_chat():
    """Return to the chat that originally opened the resource workspace."""
    origin_id = st.session_state.get(
        "resource_origin_conversation_id"
    )

    current_id = st.session_state.get(
        "current_conversation_id"
    )

    # If another saved chat was selected while the resource page was open,
    # restore the original saved conversation.
    if origin_id and origin_id != current_id:
        select_conversation(origin_id)

    else:
        # Restore the original in-session state. This also handles an
        # unsaved conversation whose ID is None.
        st.session_state.current_conversation_id = origin_id

        st.session_state.chat_history = list(
            st.session_state.get(
                "resource_origin_chat_history"
            ) or []
        )

        st.session_state.youtube_url_input = (
            st.session_state.get("resource_origin_youtube_url")
            or DEFAULT_URL
        )

        st.session_state.last_processed_url = (
            st.session_state.get(
                "resource_origin_last_processed_url"
            )
        )

        st.session_state.rag_chain = (
            st.session_state.get(
                "resource_origin_rag_chain"
            )
        )

        st.session_state.pending_chat_load = False

    st.session_state.active_page = "home"
    st.rerun()


# ============================================================
# NOTES SETTINGS DIALOG
# ============================================================

@st.dialog("Generate Study Notes", width="large")
def notes_settings_dialog():
    st.write(
        "Choose the language for your video study notes."
    )

    language = st.selectbox(
        "Notes language",
        ["English", "Hindi", "Hinglish"],
        key="notes_dialog_language",
    )

    if not video_is_ready():
        st.warning(
            "Process a YouTube video before generating study notes."
        )
        return

    if not has_user_question(st.session_state.chat_history):
        st.info(
            "Ask at least one question in Video Chat before generating notes."
        )
        return

    if st.button(
        "✨ Generate Study Notes",
        type="primary",
        use_container_width=True,
    ):
        active_video_url = st.session_state.last_processed_url

        try:
            with st.spinner(
                "Creating study notes, analyzing visuals and generating PDF..."
            ):
                result = generate_study_material(
                    youtube_url=active_video_url,
                    chat_history=st.session_state.chat_history,
                    language=language,
                )

            if not isinstance(result, tuple) or len(result) != 3:
                raise ValueError(
                    "generate_study_material() must return "
                    "(notes, diagram_images, pdf_bytes)."
                )

            notes, diagram_images, pdf_bytes = result
            notes = clean_notes_text(notes)

            if not notes:
                raise ValueError("Generated notes are empty.")

            if not isinstance(diagram_images, list):
                diagram_images = []

            if (
                not isinstance(pdf_bytes, bytes)
                or not pdf_bytes.startswith(b"%PDF")
            ):
                raise ValueError("Generated PDF is invalid.")

            st.session_state.generated_notes = notes
            st.session_state.generated_diagrams = diagram_images
            st.session_state.notes_source_url = active_video_url
            st.session_state.generated_notes_language = language
            st.session_state.generated_pdf = pdf_bytes
            st.session_state.generated_diagram_count = len(
                diagram_images
            )
            st.session_state.study_generation_error = None

            st.session_state.active_page = "notes"
            st.rerun()

        except Exception as exc:
            st.error(
                f"Study notes generation failed: {exc}"
            )


# ============================================================
# QUIZ SETTINGS DIALOG
# ============================================================

@st.dialog("Generate Quiz", width="large")
def quiz_settings_dialog():
    st.write("Customize your quiz before generating it.")

    language = st.selectbox(
        "Quiz language",
        ["English", "Hindi", "Hinglish"],
        key="quiz_dialog_language",
    )

    question_count = st.number_input(
        "Number of questions",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        key="quiz_dialog_question_count",
    )

    difficulty = st.selectbox(
        "Difficulty",
        ["Mixed", "Easy", "Medium", "Hard"],
        key="quiz_dialog_difficulty",
    )

    question_type_labels = {
        "MCQs": "mcq",
        "True / False": "true_false",
        "Assertion–Reason": "assertion_reason",
        "Multiple Select": "multiple_select",
        "Fill in the Blank": "fill_blank",
        "Scenario Based": "scenario",
    }

    selected_type_labels = st.multiselect(
        "Question formats",
        options=list(question_type_labels.keys()),
        default=list(question_type_labels.keys()),
        key="quiz_dialog_question_types",
    )

    if not video_is_ready():
        st.warning(
            "Process a YouTube video before generating a quiz."
        )
        return

    if not selected_type_labels:
        st.info("Select at least one question format.")

    if st.button(
    "🧠 Generate Quiz",
    type="primary",
    use_container_width=True,
    disabled=not selected_type_labels,
):
     try:
        active_video_url = st.session_state.last_processed_url

        normalized_video_url = normalize_youtube_url(
            active_video_url
        )

        # Retrieve previously generated questions for this video.
        history_by_video = dict(
            st.session_state.get(
                "quiz_question_history_by_video",
                {},
            )
        )

        previous_questions = list(
            history_by_video.get(
                normalized_video_url,
                [],
            )
        )

        with st.spinner(
            "Preparing fresh questions from your video, notes and chat..."
        ):
            quiz = generate_video_quiz(
                youtube_url=active_video_url,
                notes=st.session_state.generated_notes,
                chat_history=st.session_state.chat_history,
                language=language,
                question_count=int(question_count),
                difficulty=difficulty,
                question_types=[
                    question_type_labels[label]
                    for label in selected_type_labels
                ],
                previous_questions=previous_questions,
            )

        questions = validate_quiz_before_display(quiz)

        # Add newly generated questions to this video's history.
        new_question_texts = [
            str(question.get("question", "")).strip()
            for question in questions
            if str(question.get("question", "")).strip()
        ]

        updated_history = previous_questions.copy()

        for question_text in new_question_texts:
            if question_text not in updated_history:
                updated_history.append(question_text)

        # Keep the most recent 150 questions.
        history_by_video[normalized_video_url] = (
            updated_history[-150:]
        )

        st.session_state.quiz_question_history_by_video = (
            history_by_video
        )

        # A new quiz always starts with clean answers.
        clear_quiz_answers()

        st.session_state.quiz_data = quiz
        st.session_state.quiz_result = None
        st.session_state.quiz_source_url = active_video_url
        st.session_state.quiz_started_at = time.time()
        st.session_state.quiz_attempt_id = make_attempt_id()
        st.session_state.quiz_attempt_recorded_id = None
        st.session_state.quiz_question_index = 0

        st.session_state.active_page = "quiz"
        st.rerun()

     except Exception as exc:
        st.error(f"Quiz generation failed: {exc}")
# ============================================================
# HOME PAGE — HEADER, VIDEO INPUT AND CHAT
# ============================================================

if st.session_state.active_page == "home":

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # YOUTUBE INPUT
    # --------------------------------------------------------

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
        current_url = (
            st.session_state.youtube_url_input or ""
        ).strip()

        if active_video_url and active_video_url == current_url:
            st.success(
                "Video is ready. You can start asking questions below."
            )
            st.caption(f"Processed source: {active_video_url}")

        elif active_video_url and active_video_url != current_url:
            st.warning(
                "The URL has changed. Process this video before continuing."
            )

        else:
            st.caption(
                "Paste a public YouTube URL and select Process Video to begin."
            )

    # --------------------------------------------------------
    # PROCESS VIDEO
    # --------------------------------------------------------

    if process_button:
        requested_url = (
            st.session_state.youtube_url_input or ""
        ).strip()

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
                    "Loading transcript, creating embeddings, "
                    "and building the RAG chain..."
                ):
                    new_chain = initialize_rag_chain(requested_url)

                old_url = st.session_state.last_processed_url

                st.session_state.rag_chain = new_chain
                st.session_state.last_processed_url = requested_url

                if old_url != requested_url:
                    clear_study_material()
                    reset_current_quiz()

                current_id = st.session_state.current_conversation_id

                if current_id:
                    update_video_url(
                        current_id,
                        requested_url,
                    )

                st.success("Video processed successfully.")

            except Exception as exc:
                st.error(f"Video processing failed: {exc}")

    # --------------------------------------------------------
    # CHAT INTERFACE
    # --------------------------------------------------------

    with st.container(border=True):
        st.subheader("💬 Video Chat")

        st.caption(
            "Ask questions about the processed video. "
            "The answer can use the video context and relevant general knowledge."
        )

        current_id = st.session_state.current_conversation_id

        if current_id:
            st.session_state.chat_history = get_messages(current_id)
        else:
            st.session_state.chat_history = []

        # Display existing messages.
        for message in st.session_state.chat_history:
            role = message.get("role", "assistant")

            if role not in ("user", "assistant"):
                role = "assistant"

            with st.chat_message(role):
                st.markdown(message.get("content", ""))

        prompt = st.chat_input(
            "Ask a question about the video..."
        )

        if prompt:
            chain = st.session_state.rag_chain

            processed_url = st.session_state.last_processed_url
            current_url = (
                st.session_state.youtube_url_input or ""
            ).strip()

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

                add_message(current_id, "user", prompt)

                with st.chat_message("user"):
                    st.markdown(prompt)

                try:
                    with st.chat_message("assistant"):
                        with st.spinner("Searching the transcript..."):
                            raw_response = chain.invoke(prompt)

                        response = get_answer_text(raw_response)

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
                    error_message = (
                        f"Could not generate an answer: {exc}"
                    )

                    st.error(
                        f"Answer generation failed: {exc}"
                    )

                    add_message(
                        current_id,
                        "assistant",
                        error_message,
                    )

                st.rerun()

    # Refresh current conversation messages.
    current_id = st.session_state.current_conversation_id

    if current_id:
        st.session_state.chat_history = get_messages(current_id)


# ============================================================
# LEARNING RESOURCE CARDS — HOME PAGE ONLY
# ============================================================

if st.session_state.active_page == "home":

    st.markdown("### ✨ Learning Resources")

    resource_cols = st.columns(3, gap="medium")

    # Study Notes
    with resource_cols[0]:
        with st.container(border=True):
            st.markdown("#### 📝 Study Notes")
            st.caption("🟢 Available")
            st.write(
                "Create structured notes and a downloadable PDF."
            )

            if st.button(
                "Open",
                key="open_notes_settings",
                use_container_width=True,
                disabled=not video_is_ready(),
            ):
                capture_resource_origin()
                notes_settings_dialog()

    # Presentation
    with resource_cols[1]:
        with st.container(border=True):
            st.markdown("#### 📊 Presentation")
            st.caption("🟠 Coming Soon")
            st.write(
                "Turn your video into a presentation."
            )

            st.button(
                "Coming Soon",
                key="presentation_coming_soon",
                use_container_width=True,
                disabled=True,
            )

    # Quiz
    with resource_cols[2]:
        with st.container(border=True):
            st.markdown("#### 🧠 Quiz Practice")
            st.caption("🟢 Available")
            st.write(
                "Test your understanding with different question formats."
            )

            if st.button(
                "Open",
                key="open_quiz_settings",
                use_container_width=True,
                disabled=not video_is_ready(),
            ):
                capture_resource_origin()
                quiz_settings_dialog()

    if not video_is_ready():
        st.caption(
            "Process a YouTube video to enable Study Notes and Quiz Practice."
        )


# ============================================================
# SEPARATE STUDY NOTES WORKSPACE
# ============================================================

if (
    st.session_state.active_page == "notes"
    and st.session_state.generated_notes
):

    if (
        st.session_state.notes_source_url
        != st.session_state.last_processed_url
    ):
        st.warning(
            "These notes belong to another video. "
            "Return to the original chat or generate notes again."
        )

        if st.button(
            "← Back to Chat",
            key="back_from_mismatched_notes",
        ):
            return_to_originating_chat()

    else:
        back_col, title_col = st.columns(
            [1, 5],
            vertical_alignment="center",
        )

        with back_col:
            if st.button(
                "← Back to Chat",
                key="back_from_notes",
            ):
                return_to_originating_chat()

        with title_col:
            st.markdown("# 📝 Your Study Notes")

        diagram_count = st.session_state.generated_diagram_count

        visual_status = (
            f"{diagram_count} educational visual(s) included"
            if diagram_count
            else (
                "No suitable visuals were detected; "
                "notes are available without diagrams."
            )
        )

        st.caption(
            f"Language: {st.session_state.generated_notes_language} "
            f"| {visual_status}"
        )

        with st.container(border=True):
            st.markdown("### 📖 Handwritten Notes Preview")

            try:
                preview_html = create_document_html(
                    notes=st.session_state.generated_notes,
                    title="YouTube Video Study Notes",
                    language=st.session_state.generated_notes_language,
                    diagram_images=st.session_state.generated_diagrams,
                )

                components.html(
                    preview_html,
                    height=1100,
                    scrolling=True,
                )

            except Exception as exc:
                st.error(
                    f"Could not render handwritten notes preview: {exc}"
                )
                st.markdown(st.session_state.generated_notes)

        pdf_bytes = st.session_state.generated_pdf

        if pdf_bytes:
            with st.container(border=True):
                st.markdown("### 📥 Download Your Study Material")

                st.download_button(
                    label="📥 Download Complete Study Notes PDF",
                    data=pdf_bytes,
                    file_name=create_pdf_filename(),
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                    key="download_complete_study_pdf",
                )

        notes_actions = st.columns(2)

        with notes_actions[0]:
            if st.button(
                "🔄 Regenerate Study Notes",
                use_container_width=True,
                key="regenerate_notes",
            ):
                clear_study_material()
                return_to_originating_chat()

        with notes_actions[1]:
            if st.button(
                "🧹 Clear Study Notes",
                use_container_width=True,
                key="clear_notes",
            ):
                clear_study_material()
                return_to_originating_chat()


# ============================================================
# QUIZ ANSWER COLLECTION
# ============================================================

def collect_submitted_answers(questions):
    """Collect all answers from persistent quiz state."""
    submitted_answers = {}

    saved_answers = dict(
        st.session_state.get("quiz_saved_answers", {})
    )

    for question in questions:
        question_id = str(question.get("id", ""))
        question_type = question.get("type", "mcq")

        if question_type == "multiple_select":
            selected = saved_answers.get(question_id)

            # Fallback to widget state if needed.
            if selected is None:
                selected = []

                for option in question.get("options", []):
                    option_id = str(option.get("id", ""))

                    checkbox_key = (
                        f"quiz_multi_{question_id}_{option_id}"
                    )

                    if st.session_state.get(checkbox_key, False):
                        selected.append(option_id)

            submitted_answers[question_id] = list(selected)

        else:
            answer = saved_answers.get(question_id)

            # Fallback for the currently rendered widget.
            if answer is None:
                answer_key = f"quiz_answer_{question_id}"
                answer = st.session_state.get(answer_key)

            submitted_answers[question_id] = answer

    return submitted_answers
# ============================================================
# QUIZ PERFORMANCE HISTORY
# ============================================================

def record_quiz_attempt(quiz, evaluated_result):
    """
    Save one completed attempt.

    The attempt ID prevents the same submission from being added twice
    during Streamlit reruns.
    """
    attempt_id = st.session_state.get("quiz_attempt_id")

    if not attempt_id:
        attempt_id = make_attempt_id()
        st.session_state.quiz_attempt_id = attempt_id

    if (
        attempt_id
        == st.session_state.get("quiz_attempt_recorded_id")
    ):
        return

    attempt_record = {
        "attempt_id": attempt_id,
        "quiz_title": quiz.get(
            "title",
            "Video Learning Quiz",
        ),
        "source_url": st.session_state.get("quiz_source_url"),
        "completed_at": datetime.now().strftime(
            "%d %b %Y, %I:%M %p"
        ),
        "score": evaluated_result.get("score", 0),
        "total_questions": evaluated_result.get(
            "total_questions",
            0,
        ),
        "percentage": evaluated_result.get("percentage", 0),
        "performance": evaluated_result.get(
            "performance",
            "Completed",
        ),
        "time_taken": evaluated_result.get(
            "time_taken_formatted",
            "Not available",
        ),
    }

    history = list(
        st.session_state.get("quiz_attempt_history", [])
    )

    history.append(attempt_record)

    st.session_state.quiz_attempt_history = history
    st.session_state.quiz_attempt_recorded_id = attempt_id


# ============================================================
# SEPARATE QUIZ WORKSPACE — ONE QUESTION AT A TIME
# ============================================================

if (
    st.session_state.active_page == "quiz"
    and st.session_state.quiz_data
):

    quiz = st.session_state.quiz_data
    questions = quiz.get("questions", [])

    if (
        st.session_state.quiz_source_url
        != st.session_state.last_processed_url
    ):
        st.warning(
            "This quiz belongs to a different video. "
            "Return to the original chat before generating another quiz."
        )

        if st.button(
            "← Back to Chat",
            key="back_from_mismatched_quiz",
        ):
            return_to_originating_chat()

    else:
        # ----------------------------------------------------
        # QUIZ HEADER
        # ----------------------------------------------------

        back_col, heading_col = st.columns(
            [1, 5],
            vertical_alignment="center",
        )

        with back_col:
            if st.button(
                "← Back to Chat",
                key="back_from_quiz",
            ):
                return_to_originating_chat()

        with heading_col:
            st.markdown("# 🧠 Your Video Quiz")

        st.markdown(
            f"### {quiz.get('title', 'Video Learning Quiz')}"
        )

        total_questions = len(questions)

        st.caption(
            f"{total_questions} questions | "
            f"Difficulty: {quiz.get('difficulty', 'Mixed')} | "
            f"Language: {quiz.get('language', 'English')}"
        )

        st.info(
            quiz.get(
                "instructions",
                "Answer each question and submit your quiz.",
            )
        )

        result = st.session_state.quiz_result

        # ----------------------------------------------------
        # QUESTION VIEW
        # ----------------------------------------------------

        if result is None and questions:

            current_index = min(
                max(
                    int(st.session_state.quiz_question_index),
                    0,
                ),
                total_questions - 1,
            )

            question = questions[current_index]

            question_id = str(question["id"])
            question_type = question.get("type", "mcq")

            answer_key = f"quiz_answer_{question_id}"

            st.progress(
                (current_index + 1) / total_questions
            )

            st.caption(
                f"Question {current_index + 1} of {total_questions}"
            )

            with st.container(border=True):

                st.caption(
                    f"{question.get('type_label', question_type)} | "
                    f"{question.get('difficulty', 'Medium')} | "
                    f"Topic: {question.get('topic', 'General')}"
                )

                st.markdown(
                    f"### {question.get('question', '').strip()}"
                )

                options = question.get("options", [])

                # Fill in the blank
                if question_type == "fill_blank":

                    saved_answer = st.session_state.quiz_saved_answers.get(
                        question_id
                    )

                    if answer_key not in st.session_state and saved_answer is not None:
                        st.session_state[answer_key] = saved_answer

                    st.text_input(
                        "Your answer",
                        key=answer_key,
                        placeholder="Type your answer...",
                        on_change=save_quiz_answer,
                        args=(question_id, answer_key),
                    )
                # Multiple select
                # Multiple select
                elif question_type == "multiple_select":
                    st.write("**Select all correct answers:**")

                    saved_selected = set(
                        st.session_state.quiz_saved_answers.get(
                            question_id,
                            [],
                        )
                    )

                    for option in options:
                        option_id = str(option.get("id", ""))

                        checkbox_key = (
                            f"quiz_multi_{question_id}_{option_id}"
                        )

                        # Restore previously saved selection.
                        if checkbox_key not in st.session_state:
                            st.session_state[checkbox_key] = (
                                option_id in saved_selected
                            )

                        st.checkbox(
                            f"{option_id}. {option.get('text', '')}",
                            key=checkbox_key,
                            on_change=save_quiz_multi_answer,
                            args=(
                                question_id,
                                option_id,
                                checkbox_key,
                            ),
                        )

                # MCQ, True/False, Assertion-Reason and Scenario
                # MCQ, True/False, Assertion-Reason and Scenario
                else:
                    option_ids = [
                        str(option.get("id", ""))
                        for option in options
                    ]

                    option_labels = {
                        str(option.get("id", "")): (
                            f"{option.get('id', '')}. "
                            f"{option.get('text', '')}"
                        )
                        for option in options
                    }

                    saved_answer = st.session_state.quiz_saved_answers.get(
                        question_id
                    )

                    if answer_key not in st.session_state and saved_answer is not None:
                        st.session_state[answer_key] = saved_answer

                    st.radio(
                        "Choose one answer",
                        options=option_ids,
                        index=None,
                        format_func=lambda option_id: (
                            option_labels.get(
                                option_id,
                                option_id,
                            )
                        ),
                        key=answer_key,
                        on_change=save_quiz_answer,
                        args=(question_id, answer_key),
                    )
            # ------------------------------------------------
            # QUESTION NAVIGATION
            # ------------------------------------------------

            nav_left, nav_middle, nav_right = st.columns(
                [1, 2, 1]
            )

            with nav_left:
                if st.button(
                    "← Previous",
                    disabled=current_index == 0,
                    use_container_width=True,
                    key=f"quiz_previous_{current_index}",
                ):
                    st.session_state.quiz_question_index = (
                        current_index - 1
                    )
                    st.rerun()

            with nav_middle:
                st.caption(
                    "Your answers are saved as you move between questions.",
                    text_alignment="center",
                )

            with nav_right:
                if current_index < total_questions - 1:
                    if st.button(
                        "Next →",
                        type="primary",
                        use_container_width=True,
                        key=f"quiz_next_{current_index}",
                    ):
                        st.session_state.quiz_question_index = (
                            current_index + 1
                        )
                        st.rerun()

                else:
                    if st.button(
                        "✅ Submit Quiz",
                        type="primary",
                        use_container_width=True,
                        key="submit_video_quiz",
                    ):
                        submitted_answers = (
                            collect_submitted_answers(questions)
                        )

                        elapsed_seconds = (
                            int(
                                time.time() - st.session_state.quiz_started_at
                            )
                            if st.session_state.quiz_started_at
                            else 0
                        )

                        try:
                            evaluated_result = evaluate_quiz_attempt(
                                quiz=quiz,
                                submitted_answers=submitted_answers,
                                time_taken_seconds=elapsed_seconds,
                            )

                            st.session_state.quiz_result = (
                                evaluated_result
                            )

                            record_quiz_attempt(
                                quiz,
                                evaluated_result,
                            )

                            st.rerun()

                        except Exception as exc:
                            st.error(
                                f"Quiz evaluation failed: {exc}"
                            )

        # ----------------------------------------------------
        # PERFORMANCE REPORT
        # ----------------------------------------------------

        if result is not None:

            st.divider()
            st.markdown("## 📊 Your Performance Report")

            score_cols = st.columns(4)

            score_cols[0].metric(
                "Score",
                f"{result.get('score', 0)}/"
                f"{result.get('total_questions', total_questions)}",
            )

            score_cols[1].metric(
                "Percentage",
                f"{result.get('percentage', 0)}%",
            )

            score_cols[2].metric(
                "Correct",
                result.get("correct", 0),
            )

            score_cols[3].metric(
                "Incorrect / Unanswered",
                (
                    result.get("incorrect", 0)
                    + result.get("unanswered", 0)
                ),
            )

            st.subheader(
                f"Performance: {result.get('performance', 'Completed')}"
            )

            st.write(
                result.get(
                    "feedback",
                    "Quiz completed.",
                )
            )

            st.caption(
                "⏱️ Time taken: "
                + str(
                    result.get(
                        "time_taken_formatted",
                        "Not available",
                    )
                )
            )

            percentage = float(
                result.get("percentage", 0) or 0
            )

            st.progress(
                min(max(percentage / 100, 0.0), 1.0)
            )

            # -----------------------------------------------
            # ATTEMPT HISTORY
            # -----------------------------------------------

            attempt_history = st.session_state.get(
                "quiz_attempt_history",
                [],
            )

            if attempt_history:
                st.markdown("### 📈 Quiz Performance History")

                st.dataframe(
                    [
                        {
                            "Attempt": index + 1,
                            "Quiz": item.get(
                                "quiz_title",
                                "Video Learning Quiz",
                            ),
                            "Completed": item.get(
                                "completed_at",
                                "",
                            ),
                            "Score": (
                                f"{item.get('score', 0)}/"
                                f"{item.get('total_questions', 0)}"
                            ),
                            "Percentage": (
                                f"{item.get('percentage', 0)}%"
                            ),
                            "Performance": item.get(
                                "performance",
                                "",
                            ),
                        }
                        for index, item in enumerate(attempt_history)
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

            # -----------------------------------------------
            # TOPIC-WISE REPORT
            # -----------------------------------------------

            st.markdown("### 📚 Topic-wise Performance")

            topic_report = result.get("topic_report", [])

            if topic_report:
                st.dataframe(
                    topic_report,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption(
                    "Topic-wise performance is not available for this attempt."
                )

            # -----------------------------------------------
            # STRENGTHS AND WEAK TOPICS
            # -----------------------------------------------

            strength_col, weak_col = st.columns(2)

            with strength_col:
                st.markdown("#### 💪 Strong Topics")

                strengths = result.get("strengths", [])

                if strengths:
                    for topic in strengths:
                        st.success(topic)
                else:
                    st.caption(
                        "No strong topic identified yet."
                    )

            with weak_col:
                st.markdown("#### 🎯 Topics to Improve")

                weak_topics = result.get("weak_topics", [])

                if weak_topics:
                    for topic in weak_topics:
                        st.warning(topic)
                else:
                    st.success(
                        "No major weak topic identified."
                    )

            # -----------------------------------------------
            # REVISION SUGGESTIONS
            # -----------------------------------------------

            st.markdown("### 📝 Revision Suggestions")

            suggestions = result.get(
                "revision_suggestions",
                [],
            )

            if suggestions:
                for suggestion in suggestions:
                    st.markdown(f"- {suggestion}")
            else:
                st.caption(
                    "No revision suggestions were returned."
                )

            # -----------------------------------------------
            # ANSWER REVIEW
            # -----------------------------------------------

            st.markdown("## 📖 Answer Review")

            answer_results = result.get("results", [])

            for index, answer_result in enumerate(
                answer_results,
                start=1,
            ):
                status = answer_result.get("status", "unanswered")

                if status == "correct":
                    status_label = "✅ Correct"
                elif status == "incorrect":
                    status_label = "❌ Incorrect"
                else:
                    status_label = "⚪ Not Answered"

                topic = answer_result.get(
                    "topic",
                    "General",
                )

                with st.expander(
                    f"Q{index}. {status_label} — {topic}"
                ):
                    st.markdown(
                        answer_result.get("question", "")
                    )

                    submitted_answer = answer_result.get(
                        "submitted_answer"
                    )

                    if isinstance(submitted_answer, list):
                        submitted_answer = ", ".join(
                            str(item)
                            for item in submitted_answer
                        )

                    st.write(
                        "**Your answer:** "
                        + str(
                            submitted_answer
                            if submitted_answer
                            else "Not answered"
                        )
                    )

                    correct_answer = answer_result.get(
                        "correct_answer",
                        "Not available",
                    )

                    if isinstance(correct_answer, list):
                        correct_answer = ", ".join(
                            str(item)
                            for item in correct_answer
                        )

                    st.success(
                        f"Correct answer: {correct_answer}"
                    )

                    st.markdown("**Explanation:**")

                    st.write(
                        answer_result.get(
                            "explanation",
                            "No explanation was provided.",
                        )
                    )

                    evidence = answer_result.get("evidence")

                    if evidence:
                        st.caption(
                            f"Source concept: {evidence}"
                        )

            # -----------------------------------------------
            # GENERATE ANOTHER QUIZ
            # -----------------------------------------------

            if st.button(
                "🔄 Generate Another Quiz",
                type="primary",
                use_container_width=True,
                key="retake_video_quiz",
            ):
                reset_current_quiz()

                # Return to Home so the learner can open the quiz
                # settings dialog and choose new options.
                st.session_state.active_page = "home"
                st.rerun()