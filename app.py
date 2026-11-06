import os
import re
import json
import base64
import tempfile
from io import BytesIO
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import PromptTemplate

from pdf_generator import create_study_notes_pdf


# ============================================================
# OPTIONAL VIDEO / VISION DEPENDENCIES
# ============================================================

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

YOUTUBE_URL = "https://youtu.be/p4pHsuEf4Ms?t=9"
DEFAULT_YOUTUBE_URL = YOUTUBE_URL

EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "gemini-2.5-flash"

# RAG settings
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 150
RETRIEVER_K = 5

# Study notes settings
SUMMARY_CHUNK_SIZE = 7500
SUMMARY_CHUNK_OVERLAP = 250
SUMMARIES_PER_NOTES_BATCH = 3
MAX_CHAT_QUESTIONS = 15

# Video visual settings
MAX_VIDEO_FRAMES = 18
MAX_VIDEO_DOWNLOAD_MB = 180
MAX_DIAGRAMS = 5
MIN_VISUAL_CONFIDENCE = 0.65

# Image settings
MAX_FRAME_WIDTH = 1200
JPEG_QUALITY = 85


# ============================================================
# LANGUAGE MODEL
# ============================================================

def get_llm():
    """Create the Groq language model."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is missing. "
            "Add it to your .env file."
        )

    return ChatGroq(
        model=LLM_MODEL,
        temperature=0.2,
    )


def response_to_text(response):
    """Convert model responses into plain text."""

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                item_text = item.get("text")

                if item_text:
                    parts.append(str(item_text))

        return "\n".join(parts).strip()

    return str(content).strip() if content else ""


# ============================================================
# YOUTUBE TRANSCRIPT
# ============================================================

def load_transcript(youtube_url):
    """Load the available English or Hindi YouTube transcript."""

    if not youtube_url or not youtube_url.strip():
        raise ValueError("Please provide a valid YouTube URL.")

    loader = YoutubeLoader.from_youtube_url(
        youtube_url.strip(),
        add_video_info=False,
        language=["en", "hi"],
    )

    documents = loader.load()

    if not documents:
        raise ValueError(
            "No supported transcript was found. "
            "Check whether captions are available for this video."
        )

    if not any(
        document.page_content.strip()
        for document in documents
    ):
        raise ValueError("The video transcript is empty.")

    return documents


def format_docs(documents):
    return "\n\n".join(
        document.page_content
        for document in documents
    )


# ============================================================
# RAG QUESTION-ANSWERING PIPELINE
# ============================================================

def initialize_rag_chain(youtube_url):
    """Create the transcript-based FAISS retrieval chain."""

    documents = load_transcript(youtube_url)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "The transcript could not be divided into searchable sections."
        )

    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is missing. "
            "Add it to your .env file."
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )

    prompt = PromptTemplate.from_template(
        """
You are a helpful YouTube video-learning assistant.

Answer the user's question using the supplied transcript context.

Rules:
- Use the transcript as your primary source.
- Do not invent facts or quotations.
- If the answer is not supported by the transcript, say so clearly.
- Respond in the same language as the user's question.
- Use simple English, Hindi, or natural Hinglish.
- Explain technical terms clearly.
- Use examples when they help understanding.
- Structure longer answers with headings and bullet points.

Transcript context:
{context}

Question:
{question}

Answer:
"""
    )

    chain = (
        RunnableParallel({
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        })
        | prompt
        | get_llm()
        | StrOutputParser()
    )

    return chain


# ============================================================
# CHAT HISTORY HELPERS
# ============================================================

def get_user_questions(chat_history):
    if not isinstance(chat_history, list):
        return []

    return [
        str(message.get("content", "")).strip()
        for message in chat_history
        if (
            isinstance(message, dict)
            and message.get("role") == "user"
            and str(message.get("content", "")).strip()
        )
    ]


def build_questions_context(chat_history):
    questions = get_user_questions(chat_history)

    if not questions:
        return (
            "No questions were asked. "
            "Cover the complete video comprehensively."
        )

    return "\n".join(
        f"- {question}"
        for question in questions[-MAX_CHAT_QUESTIONS:]
    )


# ============================================================
# LANGUAGE INSTRUCTIONS
# ============================================================

def get_language_instruction(language):
    instructions = {
        "English": """
Write in clear, simple English.
Use standard English technical terminology.
Explain difficult terms in student-friendly language.
""",

        "Hindi": """
Write in natural Hindi using Devanagari script.
Keep standard technical terms in English where appropriate.
Avoid unnecessarily difficult Hindi vocabulary.
""",

        "Hinglish": """
Write in natural, readable Hinglish.
Use Hindi and English naturally.
Keep standard technical terms in English.
Avoid excessively informal language.
""",
    }

    if language not in instructions:
        raise ValueError(
            f"Unsupported language: {language}"
        )

    return instructions[language]


# ============================================================
# TRANSCRIPT SUMMARIZATION
# ============================================================

def summarize_transcript(documents, llm):
    """Summarize the complete transcript in manageable sections."""

    full_text = "\n\n".join(
        document.page_content.strip()
        for document in documents
        if document.page_content.strip()
    )

    if not full_text:
        raise ValueError("The transcript is empty.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SUMMARY_CHUNK_SIZE,
        chunk_overlap=SUMMARY_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(full_text)

    if not chunks:
        raise ValueError(
            "The transcript could not be divided into sections."
        )

    summaries = []

    progress = st.progress(
        0,
        text="Understanding the video transcript...",
    )

    try:
        for index, chunk in enumerate(chunks):

            prompt = f"""
You are an expert educational transcript analyst.

Summarize this part of an educational video.

Preserve:
1. Main concepts and subtopics.
2. Definitions and explanations.
3. Technical terminology.
4. Important examples.
5. Formulas, facts and numerical values.
6. Processes and individual stages.
7. Cause-and-effect relationships.
8. Differences between related concepts.
9. Important examination details.

Rules:
- Do not invent information.
- Do not add unsupported outside facts.
- Do not remove important explanations.
- Avoid unnecessary repetition.
- Preserve technical terminology.
- Do not write a conclusion for the complete video.

TRANSCRIPT PART {index + 1} OF {len(chunks)}:

{chunk}

Write a structured, information-rich summary.
"""

            response = llm.invoke(prompt)
            summary = response_to_text(response)

            if not summary:
                raise ValueError(
                    f"Could not summarize transcript part {index + 1}."
                )

            summaries.append(summary)

            progress.progress(
                (index + 1) / len(chunks),
                text=(
                    f"Understanding transcript: "
                    f"{index + 1}/{len(chunks)} sections"
                ),
            )

    finally:
        progress.empty()

    return summaries


# ============================================================
# NOTES GENERATION
# ============================================================

def generate_notes_batch(
    llm,
    summary_batch,
    batch_number,
    total_batches,
    language,
    questions_context,
):
    summary_text = "\n\n".join(
        f"VIDEO SUMMARY {index + 1}:\n{summary}"
        for index, summary in enumerate(summary_batch)
    )

    prompt = f"""
You are an expert teacher and academic study-notes writer.

Create a detailed section of study notes from the supplied summaries.

This is NOTES SECTION {batch_number} OF {total_batches}.

LANGUAGE:
{get_language_instruction(language)}

CONTENT REQUIREMENTS:
1. Cover important concepts.
2. Explain concepts clearly and accurately.
3. Include definitions and working principles.
4. Include important examples.
5. Preserve technical terminology.
6. Explain processes logically.
7. Include formulas and their meanings.
8. Include important facts and numerical values.
9. Do not invent unsupported information.
10. Avoid unnecessary repetition.

FORMATTING:
- Use Markdown headings.
- Use ## for main topics.
- Use ### for subtopics.
- Use short, readable paragraphs.
- Use bullet points where helpful.
- Use bold for important terms.
- Use Markdown tables for comparisons.
- Do not create flowcharts.
- Do not insert ASCII diagrams.
- Do not add a repeated cover title.

FORMULAS AND REACTIONS:
- Put important mathematical formulas in a separate blockquote.
- Start formula blockquotes with [FORMULA].
- Put important chemical reactions in a separate blockquote.
- Start reaction blockquotes with [REACTION].
- Explain symbols and conditions outside the formula block.

Example:

> [FORMULA] Force = Mass × Acceleration

> [REACTION] 2H₂ + O₂ → 2H₂O

IMPORTANT:
- This section will be combined with other sections.
- Do not write a general introduction unless appropriate.
- Do not write a final conclusion for the complete video.
- Avoid repeating earlier sections.
- Make this section understandable independently.

STUDENT QUESTIONS:
{questions_context}

VIDEO SUMMARIES:
{summary_text}

Generate only Markdown notes.
"""

    response = llm.invoke(prompt)
    notes = response_to_text(response)

    if not notes:
        raise ValueError(
            f"Notes generation failed for section {batch_number}."
        )

    return notes


def generate_notes_from_summaries(
    summaries,
    llm,
    language,
    chat_history,
):
    if not summaries:
        raise ValueError("No transcript summaries are available.")

    batches = [
        summaries[index:index + SUMMARIES_PER_NOTES_BATCH]
        for index in range(
            0,
            len(summaries),
            SUMMARIES_PER_NOTES_BATCH,
        )
    ]

    questions_context = build_questions_context(chat_history)
    generated_sections = []

    progress = st.progress(
        0,
        text="Preparing your study notes...",
    )

    try:
        for index, batch in enumerate(batches):

            section_notes = generate_notes_batch(
                llm=llm,
                summary_batch=batch,
                batch_number=index + 1,
                total_batches=len(batches),
                language=language,
                questions_context=questions_context,
            )

            generated_sections.append(
                section_notes.strip()
            )

            progress.progress(
                (index + 1) / len(batches),
                text=(
                    f"Writing study notes: "
                    f"{index + 1}/{len(batches)} sections"
                ),
            )

    finally:
        progress.empty()

    combined_notes = "\n\n".join(generated_sections)

    combined_notes = re.sub(
        r"\n{3,}",
        "\n\n",
        combined_notes,
    ).strip()

    if not combined_notes:
        raise ValueError(
            "The generated study notes are empty."
        )

    return combined_notes


# ============================================================
# EXTRACT VIDEO FRAMES
# ============================================================

def extract_video_frames(youtube_url):
    """
    Extract frames distributed across the video.

    These are original video frames, not AI-generated images.

    Optional dependencies:
        pip install yt-dlp opencv-python-headless

    If extraction fails, transcript-based notes still work.
    """

    if yt_dlp is None or cv2 is None:
        st.info(
            "Video visual extraction is unavailable. "
            "Install yt-dlp and opencv-python-headless "
            "to include relevant visuals from the video."
        )
        return []

    frames = []

    with tempfile.TemporaryDirectory(
        prefix="yt_visual_reference_"
    ) as temp_dir:

        video_path = os.path.join(
            temp_dir,
            "reference_video.mp4",
        )

        options = {
            "format": (
                "best[height<=480][ext=mp4]/"
                "best[height<=480]/best"
            ),
            "outtmpl": video_path,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "max_filesize": MAX_VIDEO_DOWNLOAD_MB * 1024 * 1024,
            "socket_timeout": 20,
            "retries": 1,
        }

        capture = None

        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                downloader.extract_info(
                    youtube_url,
                    download=True,
                )

            if not os.path.exists(video_path):
                candidates = list(
                    Path(temp_dir).glob("reference_video.*")
                )

                if not candidates:
                    return []

                video_path = str(candidates[0])

            capture = cv2.VideoCapture(video_path)

            if not capture.isOpened():
                return []

            frame_count = int(
                capture.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            fps = capture.get(cv2.CAP_PROP_FPS)

            if frame_count <= 0 or fps <= 0:
                return []

            duration = frame_count / fps

            # Aim for approximately one frame per minute,
            # with a maximum of MAX_VIDEO_FRAMES.
            sample_count = min(
                MAX_VIDEO_FRAMES,
                max(1, int(duration // 60)),
            )

            if sample_count == 1:
                timestamps = [duration * 0.5]
            else:
                timestamps = [
                    duration * (index + 1) / (sample_count + 1)
                    for index in range(sample_count)
                ]

            for timestamp in timestamps:

                capture.set(
                    cv2.CAP_PROP_POS_MSEC,
                    timestamp * 1000,
                )

                success, frame = capture.read()

                if not success:
                    continue

                height, width = frame.shape[:2]

                scale = min(
                    1.0,
                    MAX_FRAME_WIDTH / max(width, height),
                )

                if scale < 1:
                    frame = cv2.resize(
                        frame,
                        (
                            max(1, int(width * scale)),
                            max(1, int(height * scale)),
                        ),
                    )

                success, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [
                        int(cv2.IMWRITE_JPEG_QUALITY),
                        JPEG_QUALITY,
                    ],
                )

                if success:
                    frames.append({
                        "timestamp": float(timestamp),
                        "image": encoded.tobytes(),
                    })

        except Exception as exc:
            st.warning(
                "Video-frame extraction could not be completed. "
                "Transcript-based notes will continue. "
                f"Details: {exc}"
            )
            return []

        finally:
            if capture is not None:
                capture.release()

    return frames


# ============================================================
# VIDEO FRAME ANALYSIS
# ============================================================

def extract_json_from_response(text):
    """Extract a JSON object or array from model output."""

    if not text:
        return None

    cleaned = text.strip()

    cleaned = re.sub(
        r"\A```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```\Z",
        "",
        cleaned,
    ).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start_candidates = [
        position
        for position in (
            cleaned.find("{"),
            cleaned.find("["),
        )
        if position >= 0
    ]

    if not start_candidates:
        return None

    start = min(start_candidates)

    for end in range(len(cleaned), start, -1):
        candidate = cleaned[start:end].strip()

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def extract_note_headings(notes):
    """Get the main topic headings from generated notes."""

    headings = []

    for line in notes.splitlines():
        match = re.match(
            r"^#{2,4}\s+(.+?)\s*$",
            line.strip(),
        )

        if match:
            heading = re.sub(
                r"[*_`]",
                "",
                match.group(1),
            ).strip()

            if heading:
                headings.append(heading)

    return headings


def analyze_video_frames(frames, notes):
    """
    Analyse frames using Gemini Vision.

    The model identifies which original video frames contain
    useful educational visuals. It does not draw or generate
    scientific diagrams.
    """

    if not frames:
        return []

    if ChatGoogleGenerativeAI is None:
        st.warning(
            "Gemini Vision is unavailable. "
            "The PDF will be generated without video diagrams."
        )
        return []

    if not os.getenv("GOOGLE_API_KEY"):
        st.warning(
            "GOOGLE_API_KEY is missing. "
            "The PDF will be generated without video diagrams."
        )
        return []

    headings = extract_note_headings(notes)

    headings_text = "\n".join(
        f"- {heading}"
        for heading in headings[:60]
    )

    try:
        vision_llm = ChatGoogleGenerativeAI(
            model=VISION_MODEL,
            temperature=0.1,
        )

        content = [{
            "type": "text",
            "text": f"""
You are analysing video frames for an educational study-notes PDF.

The notes cover these topics:

{headings_text}

Your task is to identify frames that contain a genuinely useful
educational visual related to these topics.

A useful visual may be:
- A labelled scientific diagram.
- A biological structure or specimen illustration.
- A molecular or chemical structure.
- A technical architecture or component diagram.
- A graph, chart, or meaningful educational illustration.
- A process illustration that contains meaningful visual detail.

Do NOT select:
- A lecturer's face or talking-head shot.
- A plain background.
- A random video thumbnail.
- A frame that only contains subtitles.
- A frame unrelated to the study topics.
- A frame where the visual is too small or unclear to be useful.

IMPORTANT:
- Analyse only what is actually visible.
- Never invent labels or structures.
- Do not infer hidden parts of a diagram.
- Do not claim that unreadable text is readable.
- A slide containing a real diagram may be selected.
- A plain text slide should normally be rejected.
- Prefer a clear visual over a decorative slide.

Return valid JSON only, in this exact structure:

{{
  "frames": [
    {{
      "frame_number": 1,
      "is_useful": true,
      "confidence": 0.90,
      "visual_type": "scientific_diagram",
      "related_heading": "Relevant heading from the notes",
      "description": "Brief description of the visible visual",
      "visible_labels": ["Label 1", "Label 2"],
      "reason": "Why this frame is useful"
    }}
  ]
}}

Use the actual frame_number given before each image.
confidence must be between 0 and 1.
If no frame is useful, return:
{{"frames": []}}
""",
        }]

        for index, frame in enumerate(frames, start=1):

            encoded = base64.b64encode(
                frame["image"]
            ).decode("utf-8")

            content.append({
                "type": "text",
                "text": (
                    f"FRAME_NUMBER: {index}\n"
                    f"TIMESTAMP_SECONDS: {frame['timestamp']:.2f}"
                ),
            })

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:image/jpeg;base64,{encoded}"
                    )
                },
            })

        from langchain_core.messages import HumanMessage

        response = vision_llm.invoke([
            HumanMessage(content=content)
        ])

        result = extract_json_from_response(
            response_to_text(response)
        )

        if not isinstance(result, dict):
            return []

        analysed_frames = result.get("frames", [])

        if not isinstance(analysed_frames, list):
            return []

        return analysed_frames

    except Exception as exc:
        st.warning(
            "Gemini visual analysis could not be completed. "
            f"Notes will be generated without video diagrams. Details: {exc}"
        )
        return []


# ============================================================
# SELECT RELEVANT VIDEO VISUALS
# ============================================================

def timestamp_to_text(seconds):
    """Convert seconds into HH:MM:SS format."""

    seconds = max(0, int(seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{remaining_seconds:02d}"
    )


def convert_frame_to_png(image_bytes):
    """Convert the original extracted frame into PNG bytes."""

    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("RGB")

        output = BytesIO()

        image.save(
            output,
            format="PNG",
            optimize=True,
        )

        return output.getvalue()


def select_relevant_video_visuals(
    frames,
    analysis,
):
    """
    Select only frames identified as useful by Gemini.

    Original video frames are retained. No synthetic
    scientific structures or generic flowcharts are drawn.
    """

    if not frames or not analysis:
        return []

    selected = []
    used_frame_numbers = set()

    for item in analysis:

        if not isinstance(item, dict):
            continue

        if item.get("is_useful") is not True:
            continue

        try:
            confidence = float(
                item.get("confidence", 0)
            )
        except (TypeError, ValueError):
            confidence = 0

        if confidence < MIN_VISUAL_CONFIDENCE:
            continue

        try:
            frame_number = int(
                item.get("frame_number", 0)
            )
        except (TypeError, ValueError):
            continue

        if (
            frame_number < 1
            or frame_number > len(frames)
            or frame_number in used_frame_numbers
        ):
            continue

        visual_type = str(
            item.get("visual_type", "")
        ).strip().lower()

        rejected_types = {
            "talking_head",
            "person",
            "subtitle",
            "text_only",
            "irrelevant",
            "unclear",
        }

        if visual_type in rejected_types:
            continue

        frame = frames[frame_number - 1]

        try:
            png_bytes = convert_frame_to_png(
                frame["image"]
            )
        except Exception:
            continue

        title = str(
            item.get(
                "related_heading",
                "Video visual",
            )
        ).strip()

        description = str(
            item.get("description", "")
        ).strip()

        visible_labels = item.get(
            "visible_labels",
            [],
        )

        if not isinstance(visible_labels, list):
            visible_labels = []

        visible_labels = [
            str(label).strip()
            for label in visible_labels
            if str(label).strip()
        ][:12]

        selected.append({
            "number": len(selected) + 1,
            "title": title[:150],
            "related_heading": title[:150],
            "image": png_bytes,
            "timestamp": frame["timestamp"],
            "timestamp_label": timestamp_to_text(
                frame["timestamp"]
            ),
            "description": description[:500],
            "visible_labels": visible_labels,
            "visual_type": visual_type,
            "confidence": confidence,
            "source": "Original YouTube video frame",
        })

        used_frame_numbers.add(frame_number)

        if len(selected) >= MAX_DIAGRAMS:
            break

    return selected


# ============================================================
# TITLE AND PDF FILENAME
# ============================================================

def extract_notes_title(notes):
    if not isinstance(notes, str):
        return "Video Study Notes"

    for line in notes.splitlines():

        line = line.strip()

        match = re.match(
            r"^#{1,2}\s+(.+)$",
            line,
        )

        if match:
            title = re.sub(
                r"[*_`]",
                "",
                match.group(1),
            ).strip()

            if title:
                return title[:100]

    return "Video Study Notes"


def create_pdf_filename(title):
    title = re.sub(
        r"[^\w\s-]",
        "",
        str(title),
    )

    title = re.sub(
        r"[\s-]+",
        "_",
        title.strip(),
    )

    title = title.strip("_") or "Video_Study_Notes"

    return f"{title[:90]}_Notes.pdf"


# ============================================================
# COMPATIBILITY FUNCTIONS
# ============================================================

def parse_diagram_material(raw_output):
    """
    Kept for compatibility with older code.

    Synthetic diagram specifications are intentionally no
    longer used in the video-visual workflow.
    """
    return []


def generate_diagram_specifications(
    llm,
    notes,
    language,
    visual_context="",
):
    """
    Compatibility function.

    The updated workflow uses original video frames instead
    of generated generic flowcharts.
    """
    return []


def render_diagrams(diagrams):
    """
    Compatibility function.

    The updated workflow does not draw synthetic diagrams.
    """
    return diagrams if isinstance(diagrams, list) else []


# ============================================================
# GENERATE COMPLETE STUDY MATERIAL
# ============================================================

def generate_study_material(
    youtube_url,
    chat_history,
    language="English",
):
    """
    Generate complete notes and a notebook-style PDF.

    Workflow:
    1. Load transcript.
    2. Summarize the transcript.
    3. Generate detailed study notes.
    4. Extract original video frames.
    5. Analyse frames using Gemini Vision.
    6. Select relevant educational visuals.
    7. Generate the PDF.
    """

    supported_languages = {
        "English",
        "Hindi",
        "Hinglish",
    }

    if language not in supported_languages:
        raise ValueError(
            f"Unsupported language: {language}"
        )

    if not youtube_url or not youtube_url.strip():
        raise ValueError("Please provide a YouTube URL.")

    # STEP 1: Load transcript.
    with st.spinner(
        "Loading the YouTube transcript..."
    ):
        documents = load_transcript(youtube_url)

    # STEP 2: Initialize language model.
    llm = get_llm()

    # STEP 3: Summarize transcript.
    summaries = summarize_transcript(
        documents,
        llm,
    )

    # STEP 4: Generate study notes.
    notes = generate_notes_from_summaries(
        summaries=summaries,
        llm=llm,
        language=language,
        chat_history=chat_history,
    )

    if not notes.strip():
        raise ValueError(
            "The generated study notes are empty."
        )

    # STEP 5: Extract original video frames.
    with st.spinner(
        "Extracting visual references from the video..."
    ):
        frames = extract_video_frames(youtube_url)

    # STEP 6: Analyse the extracted frames.
    visual_analysis = []

    if frames:
        with st.spinner(
            f"Analysing {len(frames)} video frames..."
        ):
            visual_analysis = analyze_video_frames(
                frames=frames,
                notes=notes,
            )

    # STEP 7: Select relevant visuals.
    with st.spinner(
        "Selecting relevant educational visuals..."
    ):
        selected_visuals = select_relevant_video_visuals(
            frames=frames,
            analysis=visual_analysis,
        )

    # STEP 8: Generate PDF.
    notes_title = extract_notes_title(notes)

    with st.spinner(
        "Preparing your notebook-style PDF..."
    ):
        pdf_bytes = create_study_notes_pdf(
            notes=notes,
            title=notes_title,
            language=language,
            diagram_images=selected_visuals,
        )

    if (
        not isinstance(pdf_bytes, bytes)
        or not pdf_bytes.startswith(b"%PDF")
    ):
        raise ValueError(
            "The PDF generator did not return a valid PDF."
        )

    return (
        notes,
        selected_visuals,
        pdf_bytes,
    )


# ============================================================
# COMPATIBILITY ALIAS
# ============================================================

def generate_full_video_notes(
    youtube_url,
    chat_history,
    language="English",
):
    return generate_study_material(
        youtube_url=youtube_url,
        chat_history=chat_history,
        language=language,
    )


# ============================================================
# STANDALONE STREAMLIT APPLICATION
# ============================================================

def main():

    st.set_page_config(
        page_title="YT-RAG-Assist",
        page_icon="📚",
        layout="wide",
    )

    defaults = {
        "rag_chain": None,
        "loaded_url": None,
        "chat_history": [],
        "generated_pdf": None,
        "generated_notes": None,
        "generated_diagrams": [],
        "generated_pdf_filename": "Video_Study_Notes.pdf",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    st.title("📚 YT-RAG-Assist")

    st.write(
        "Ask questions about a YouTube video and generate "
        "complete study notes with relevant video visuals."
    )

    youtube_url = st.text_input(
        "YouTube video URL",
        value=DEFAULT_YOUTUBE_URL,
    ).strip()

    language = st.selectbox(
        "Study notes language",
        ["English", "Hindi", "Hinglish"],
    )

    if not youtube_url:
        st.warning("Enter a YouTube URL to continue.")
        return

    # Reset data when the video URL changes.
    if (
        st.session_state.loaded_url is not None
        and st.session_state.loaded_url != youtube_url
    ):
        st.session_state.rag_chain = None
        st.session_state.loaded_url = None
        st.session_state.chat_history = []
        st.session_state.generated_pdf = None
        st.session_state.generated_notes = None
        st.session_state.generated_diagrams = []

    # Load transcript for RAG.
    if st.button(
        "Load video for questions",
        type="secondary",
    ):
        try:
            with st.spinner(
                "Building the transcript search index..."
            ):
                chain = initialize_rag_chain(youtube_url)

            st.session_state.rag_chain = chain
            st.session_state.loaded_url = youtube_url
            st.session_state.chat_history = []
            st.session_state.generated_pdf = None
            st.session_state.generated_notes = None
            st.session_state.generated_diagrams = []

            st.success("Video loaded successfully.")

        except Exception as exc:
            st.error(
                f"Could not load the video: {exc}"
            )

    # Chat interface.
    if (
        st.session_state.rag_chain is not None
        and st.session_state.loaded_url == youtube_url
    ):

        st.subheader("Ask questions")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input(
            "Ask something about the video..."
        )

        if question:

            with st.chat_message("user"):
                st.markdown(question)

            try:
                with st.chat_message("assistant"):

                    with st.spinner(
                        "Searching the transcript..."
                    ):
                        answer = st.session_state.rag_chain.invoke(
                            question
                        )

                    st.markdown(answer)

                st.session_state.chat_history.extend([
                    {
                        "role": "user",
                        "content": question,
                    },
                    {
                        "role": "assistant",
                        "content": answer,
                    },
                ])

            except Exception as exc:
                st.error(
                    f"Could not answer the question: {exc}"
                )

    # Study material generation.
    st.divider()

    st.subheader("Generate complete study material")

    st.write(
        "Create detailed notes and include useful educational "
        "visuals taken from the actual video."
    )

    if st.button(
        "Generate Notes + Diagrams PDF",
        type="primary",
    ):

        try:
            notes, diagrams, pdf_bytes = generate_study_material(
                youtube_url=youtube_url,
                chat_history=st.session_state.chat_history,
                language=language,
            )

            notes_title = extract_notes_title(notes)

            st.session_state.generated_pdf = pdf_bytes
            st.session_state.generated_notes = notes
            st.session_state.generated_diagrams = diagrams

            st.session_state.generated_pdf_filename = (
                create_pdf_filename(notes_title)
            )

            st.success(
                "Study material generated successfully. "
                f"{len(diagrams)} relevant video visual(s) included."
            )

        except Exception as exc:
            st.error(
                f"Study material generation failed: {exc}"
            )

    # Notes preview.
    if st.session_state.generated_notes:

        st.divider()
        st.subheader("Study notes preview")

        st.markdown(
            st.session_state.generated_notes
        )

        if st.session_state.generated_diagrams:

            st.subheader("Relevant video visuals")

            for diagram in st.session_state.generated_diagrams:

                st.markdown(
                    f"**{diagram['title']}**"
                )

                if diagram.get("timestamp_label"):
                    st.caption(
                        "Video timestamp: "
                        f"{diagram['timestamp_label']}"
                    )

                if diagram.get("description"):
                    st.caption(
                        diagram["description"]
                    )

                st.image(
                    diagram["image"],
                    use_container_width=True,
                )

        else:
            st.info(
                "No sufficiently clear and relevant educational "
                "visuals were found in the sampled video frames. "
                "The notes PDF was generated without unrelated diagrams."
            )

    # PDF download.
    if st.session_state.generated_pdf:

        st.download_button(
            label="Download Complete Study Notes PDF",
            data=st.session_state.generated_pdf,
            file_name=st.session_state.generated_pdf_filename,
            mime="application/pdf",
            type="primary",
        )


if __name__ == "__main__":
    main()