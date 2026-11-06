import os
import re
import json
import math
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
# OPTIONAL DEPENDENCIES
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

try:
    from google import genai
except ImportError:
    genai = None


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

YOUTUBE_URL = "https://youtu.be/p4pHsuEf4Ms?t=9"
DEFAULT_YOUTUBE_URL = YOUTUBE_URL

EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "gemini-3.8-flash"
IMAGE_MODEL = "gemini-3.1-flash-image"

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 150
RETRIEVER_K = 5

SUMMARY_CHUNK_SIZE = 4000
SUMMARY_CHUNK_OVERLAP = 120
SUMMARIES_PER_NOTES_BATCH = 1

MAX_CHAT_QUESTIONS = 15

MAX_VIDEO_FRAMES = 18
MAX_VIDEO_DOWNLOAD_MB = 180
MAX_FRAME_WIDTH = 1200
JPEG_QUALITY = 85

MAX_DIAGRAMS = 5
MAX_GENERATED_DIAGRAMS = 4
MIN_VISUAL_CONFIDENCE = 0.65


# ============================================================
# LANGUAGE MODEL
# ============================================================

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    return ChatGroq(
        model=LLM_MODEL,
        temperature=0.2,
    )


def response_to_text(response):
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))

        return "\n".join(parts).strip()

    return str(content).strip() if content else ""


# ============================================================
# JSON UTILITIES
# ============================================================

def extract_json_from_response(text):
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

    starts = [
        position
        for position in (
            cleaned.find("{"),
            cleaned.find("["),
        )
        if position >= 0
    ]

    if not starts:
        return None

    start = min(starts)

    for end in range(len(cleaned), start, -1):
        try:
            return json.loads(cleaned[start:end])

        except json.JSONDecodeError:
            continue

    return None


def clean_string_list(values, limit=12):
    if not isinstance(values, list):
        return []

    return [
        str(value).strip()
        for value in values
        if value is not None and str(value).strip()
    ][:limit]


# ============================================================
# YOUTUBE TRANSCRIPT
# ============================================================

def load_transcript(youtube_url):
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
            "Check whether captions are available."
        )

    if not any(doc.page_content.strip() for doc in documents):
        raise ValueError("The video transcript is empty.")

    return documents


def format_docs(documents):
    return "\n\n".join(
        document.page_content
        for document in documents
    )


# ============================================================
# RAG PIPELINE
# ============================================================

def initialize_rag_chain(youtube_url):
    documents = load_transcript(youtube_url)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "The transcript has no searchable sections."
        )

    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY is missing.")

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

Answer using the supplied transcript context.

Rules:
- Use the transcript as your primary source.
- Do not invent facts or quotations.
- If the answer is not supported, say so.
- Respond in the same language as the question.
- Use simple English, Hindi, or natural Hinglish.
- Explain technical terms clearly.
- Use examples where helpful.
- Structure longer answers with headings and bullets.

Transcript context:
{context}

Question:
{question}

Answer:
"""
    )

    return (
        RunnableParallel({
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        })
        | prompt
        | get_llm()
        | StrOutputParser()
    )


# ============================================================
# CHAT HISTORY
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
        return "No questions were asked. Cover the complete video."

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
Use standard technical terminology.
Explain difficult terms in student-friendly language.
""",

        "Hindi": """
Write in natural Hindi using Devanagari script.
Keep standard technical terms in English where appropriate.
""",

        "Hinglish": """
Write in natural, readable Hinglish.
Use Hindi and English naturally.
Keep standard technical terms in English.
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
    full_text = "\n\n".join(
        doc.page_content.strip()
        for doc in documents
        if doc.page_content.strip()
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
            "Could not divide the transcript."
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
5. Formulas and numerical values.
6. Processes and stages.
7. Cause-and-effect relationships.
8. Differences between related concepts.
9. Important examination details.

Do not invent information.
Do not remove important explanations.
Avoid unnecessary repetition.
Do not write a conclusion for the complete video.

TRANSCRIPT PART {index + 1} OF {len(chunks)}:

{chunk}

Write a structured, information-rich summary.
"""

            response = llm.invoke(prompt)
            summary = response_to_text(response)

            if not summary:
                raise ValueError(
                    f"Could not summarize section {index + 1}."
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

Create a detailed section of study notes.

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
- Use bullets where helpful.
- Use bold for important terms.
- Use Markdown tables for comparisons.
- Do not create flowcharts.
- Do not insert ASCII diagrams.
- Do not add a repeated cover title.

FORMULAS AND REACTIONS:
- Put important formulas in a blockquote starting with [FORMULA].
- Put important chemical reactions in a blockquote starting with [REACTION].
- Explain symbols and conditions outside the formula block.

Example:

> [FORMULA] Force = Mass × Acceleration

> [REACTION] 2H₂ + O₂ → 2H₂O

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
        raise ValueError(
            "No transcript summaries are available."
        )

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

            section = generate_notes_batch(
                llm=llm,
                summary_batch=batch,
                batch_number=index + 1,
                total_batches=len(batches),
                language=language,
                questions_context=questions_context,
            )

            generated_sections.append(section.strip())

            progress.progress(
                (index + 1) / len(batches),
                text=(
                    f"Writing notes: "
                    f"{index + 1}/{len(batches)} sections"
                ),
            )

    finally:
        progress.empty()

    notes = "\n\n".join(generated_sections)
    notes = re.sub(r"\n{3,}", "\n\n", notes).strip()

    if not notes:
        raise ValueError("Generated notes are empty.")

    return notes


# ============================================================
# NOTE HEADINGS
# ============================================================

def extract_note_headings(notes):
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


# ============================================================
# VIDEO FRAME EXTRACTION
# ============================================================

def extract_video_frames(youtube_url):
    """
    Download a YouTube video and sample frames across its duration.

    Uses Deno explicitly for YouTube JavaScript challenge solving.
    Prefers progressive MP4 formats compatible with OpenCV.
    """

    if yt_dlp is None:
        st.warning(
            "yt-dlp is not installed. "
            "Original video visuals cannot be extracted."
        )
        return []

    if cv2 is None:
        st.warning(
            "OpenCV is not installed. "
            "Original video visuals cannot be extracted."
        )
        return []

    frames = []

    # --------------------------------------------------------
    # DENO CONFIGURATION
    # --------------------------------------------------------

    deno_path = (
        Path.home() / ".deno" / "bin" / "deno.exe"
    )

    if not deno_path.exists():
        st.warning(
            "Deno executable was not found. "
            "Please verify the Deno installation."
        )
        return []

    # --------------------------------------------------------
    # TEMPORARY VIDEO DOWNLOAD
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory(
        prefix="yt_visual_reference_"
    ) as temp_dir:

        video_template = os.path.join(
            temp_dir,
            "reference_video.%(ext)s",
        )

        options = {
            # Prefer progressive MP4 with audio.
            # Fall back to video-only MP4 if necessary.
            "format": (
                "18/"
                "best[height<=480][ext=mp4]/"
                "best[height<=480][ext=mp4]/"
                "best[height<=480]/best"
            ),

            "outtmpl": video_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,

            "max_filesize": (
                MAX_VIDEO_DOWNLOAD_MB * 1024 * 1024
            ),

            "socket_timeout": 30,
            "retries": 3,
            "fragment_retries": 3,
            "overwrites": True,

            # Explicit Deno executable path.
            "js_runtimes": {
                "deno": {
                    "path": str(deno_path),
                },
            },
        }

        capture = None

        try:

            # ------------------------------------------------
            # DOWNLOAD VIDEO
            # ------------------------------------------------

            with yt_dlp.YoutubeDL(options) as downloader:
                downloader.extract_info(
                    youtube_url,
                    download=True,
                )

            candidates = [
                path
                for path in Path(temp_dir).glob("reference_video.*")
                if path.is_file()
            ]

            if not candidates:
                st.warning(
                    "YouTube download completed, "
                    "but no video file was found."
                )
                return []

            # Prefer MP4 when multiple files exist.
            candidates.sort(
                key=lambda path: (
                    path.suffix.lower() != ".mp4"
                )
            )

            video_path = str(candidates[0])

            # ------------------------------------------------
            # OPEN VIDEO WITH OPENCV
            # ------------------------------------------------

            capture = cv2.VideoCapture(video_path)

            if not capture.isOpened():
                st.warning(
                    "OpenCV could not open the downloaded video. "
                    "The selected format may not be supported."
                )
                return []

            frame_count = int(
                capture.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            fps = capture.get(cv2.CAP_PROP_FPS)

            if frame_count <= 0 or fps <= 0:
                st.warning(
                    "Video duration or frame information "
                    "is unavailable."
                )
                return []

            duration = frame_count / fps

            # ------------------------------------------------
            # FRAME SAMPLING
            # ------------------------------------------------

            sample_count = min(
                MAX_VIDEO_FRAMES,
                frame_count,
                max(6, math.ceil(duration / 60)),
            )

            if sample_count <= 0:
                return []

            timestamps = [
                duration * (index + 1) / (sample_count + 1)
                for index in range(sample_count)
            ]

            # ------------------------------------------------
            # FRAME EXTRACTION
            # ------------------------------------------------

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
                "Video-frame extraction failed. "
                "Continuing with transcript notes and "
                "AI-generated illustrations where possible. "
                f"Details: {exc}"
            )

            return []

        finally:

            if capture is not None:
                capture.release()

    if frames:
        st.success(
            f"Extracted {len(frames)} original video frames."
        )
    else:
        st.warning(
            "No readable video frames were extracted."
        )

    return frames
# ============================================================
# VIDEO FRAME UTILITIES
# ============================================================

def timestamp_to_text(seconds):
    seconds = max(0, int(seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{remaining:02d}"


def convert_frame_to_png(image_bytes):
    with Image.open(BytesIO(image_bytes)) as image:
        output = BytesIO()

        image.convert("RGB").save(
            output,
            format="PNG",
            optimize=True,
        )

        return output.getvalue()


# ============================================================
# VIDEO FRAME ANALYSIS
# ============================================================

def analyze_video_frames(frames, notes):
    if not frames:
        return []

    if ChatGoogleGenerativeAI is None:
        st.warning(
            "Gemini Vision is unavailable. "
            "Original video frames will not be analysed."
        )
        return []

    if not os.getenv("GOOGLE_API_KEY"):
        st.warning(
            "GOOGLE_API_KEY is missing. "
            "Video visual analysis is disabled."
        )
        return []

    headings = extract_note_headings(notes)

    content = [{
        "type": "text",
        "text": f"""
Analyse these video frames for an educational study-notes PDF.

Study topics:
{chr(10).join("- " + heading for heading in headings[:60])}

Identify frames containing genuinely useful educational visuals:
- Labelled scientific diagrams.
- Biological structures.
- Molecular or chemical structures.
- Technical illustrations.
- Graphs and charts.
- Detailed educational drawings.

Reject:
- Talking-head shots.
- Plain text slides.
- Unrelated footage.
- Unreadable or tiny visuals.
- Decorative thumbnails.

Only describe what is actually visible.
Never invent labels or hidden structures.

Return JSON only:

{{
  "frames": [
    {{
      "frame_number": 1,
      "is_useful": true,
      "confidence": 0.90,
      "visual_type": "scientific_diagram",
      "related_heading": "Relevant notes heading",
      "description": "What is visibly shown",
      "visible_labels": ["Label 1", "Label 2"]
    }}
  ]
}}

If no frame is useful, return {{"frames": []}}.
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
                "url": f"data:image/jpeg;base64,{encoded}"
            },
        })

    try:
        from langchain_core.messages import HumanMessage

        vision_llm = ChatGoogleGenerativeAI(
            model=VISION_MODEL,
            temperature=0.1,
        )

        response = vision_llm.invoke([
            HumanMessage(content=content)
        ])

        result = extract_json_from_response(
            response_to_text(response)
        )

        if isinstance(result, dict):
            frames_result = result.get("frames", [])

            if isinstance(frames_result, list):
                return frames_result

        return []

    except Exception as exc:
        st.warning(
            f"Video visual analysis failed: {exc}"
        )
        return []


# ============================================================
# SELECT ORIGINAL VIDEO VISUALS
# ============================================================

def select_relevant_video_visuals(frames, analysis):
    if not frames or not analysis:
        return []

    selected = []
    used = set()

    rejected_types = {
        "talking_head",
        "person",
        "subtitle",
        "text_only",
        "irrelevant",
        "unclear",
    }

    for item in analysis:

        if not isinstance(item, dict):
            continue

        if item.get("is_useful") is not True:
            continue

        try:
            confidence = float(
                item.get("confidence", 0)
            )

            frame_number = int(
                item.get("frame_number", 0)
            )

        except (TypeError, ValueError):
            continue

        if confidence < MIN_VISUAL_CONFIDENCE:
            continue

        if (
            frame_number < 1
            or frame_number > len(frames)
            or frame_number in used
        ):
            continue

        visual_type = str(
            item.get("visual_type", "")
        ).strip().lower()

        if visual_type in rejected_types:
            continue

        frame = frames[frame_number - 1]

        try:
            png_bytes = convert_frame_to_png(
                frame["image"]
            )

        except Exception:
            continue

        heading = str(
            item.get("related_heading", "Video visual")
        ).strip()

        labels = clean_string_list(
            item.get("visible_labels", []),
            limit=12,
        )

        selected.append({
            "number": len(selected) + 1,
            "title": heading[:150],
            "related_heading": heading[:150],
            "image": png_bytes,
            "timestamp": frame["timestamp"],
            "timestamp_label": timestamp_to_text(
                frame["timestamp"]
            ),
            "description": str(
                item.get("description", "")
            )[:500],
            "visible_labels": labels,
            "visual_type": visual_type,
            "confidence": confidence,
            "source": "Original YouTube video frame",
        })

        used.add(frame_number)

        if len(selected) >= MAX_DIAGRAMS:
            break

    return selected


# ============================================================
# DIAGRAM SPECIFICATION GENERATION
# ============================================================

def generate_diagram_specifications(
    llm,
    notes,
    language,
    visual_context="",
):
    """
    Identify concepts that need proper illustrations.

    This function generates specifications, not image files.
    """

    headings = extract_note_headings(notes)

    prompt = f"""
You are an educational science and technical diagram planner.

Read the study notes and identify concepts that genuinely benefit
from a detailed labelled illustration.

NOTES:
{notes[:18000]}

AVAILABLE VIDEO VISUALS:
{visual_context or "No useful video visuals were detected."}

MAIN NOTE HEADINGS:
{json.dumps(headings, ensure_ascii=False)}

Create up to {MAX_GENERATED_DIAGRAMS} diagram specifications.

Prioritize:
1. Important biological structures.
2. Scientific structures and components.
3. Molecular or chemical structures.
4. Technical components and architectures.
5. Important mechanisms that need a real illustration.

Do not create generic flowcharts.
Do not create decorative diagrams.
Do not repeat a concept already clearly shown in an available video visual.
Do not create diagrams for simple definitions that need no illustration.

For each diagram, provide:
- title
- related_heading: exact or closest heading from notes
- subject
- diagram_type
- key_components
- relationships
- visible_labels
- purpose
- source_basis

Rules:
- Use only facts supported by the notes.
- Do not add unsupported organelles, structures or mechanisms.
- Do not invent numerical values.
- Keep each diagram focused on one concept.
- Prefer 4 to 10 meaningful labels.
- Do not assume every topic needs a cell diagram.
- Select diagrams based on the actual subject of the notes.

Return JSON only:

{{
  "diagrams": [
    {{
      "title": "Diagram title",
      "related_heading": "Notes heading",
      "subject": "What should be illustrated",
      "diagram_type": "scientific_structure",
      "key_components": ["Component 1", "Component 2"],
      "relationships": ["Component relationship"],
      "visible_labels": ["Label 1", "Label 2"],
      "purpose": "Why this diagram is useful",
      "source_basis": "Supporting explanation from the notes"
    }}
  ]
}}

If no diagram is needed, return {{"diagrams": []}}.
"""

    response = llm.invoke(prompt)

    result = extract_json_from_response(
        response_to_text(response)
    )

    if not isinstance(result, dict):
        return []

    diagrams = result.get("diagrams", [])

    if not isinstance(diagrams, list):
        return []

    valid = []

    for item in diagrams:

        if not isinstance(item, dict):
            continue

        required = (
            "title",
            "related_heading",
            "subject",
            "key_components",
        )

        if not all(item.get(key) for key in required):
            continue

        components = item.get("key_components")

        if not isinstance(components, list) or not components:
            continue

        item["key_components"] = clean_string_list(
            components,
            limit=12,
        )

        item["relationships"] = clean_string_list(
            item.get("relationships", []),
            limit=12,
        )

        item["visible_labels"] = clean_string_list(
            item.get("visible_labels", []),
            limit=12,
        )

        if not item["key_components"]:
            continue

        valid.append(item)

        if len(valid) >= MAX_GENERATED_DIAGRAMS:
            break

    return valid


# ============================================================
# IMAGE GENERATION PROMPT
# ============================================================

def generate_scientific_image_prompt(
    specification,
    language="English",
):
    title = specification["title"]
    subject = specification["subject"]

    components = specification.get(
        "key_components",
        [],
    )

    relationships = specification.get(
        "relationships",
        [],
    )

    labels = specification.get(
        "visible_labels",
        [],
    )

    return f"""
Create a high-quality educational scientific illustration
for a student's study-notes PDF.

SUBJECT:
{subject}

TITLE:
{title}

REQUIRED COMPONENTS:
{json.dumps(components, ensure_ascii=False)}

REQUIRED RELATIONSHIPS:
{json.dumps(relationships, ensure_ascii=False)}

EXACT LABELS TO INCLUDE:
{json.dumps(labels, ensure_ascii=False)}

VISUAL STYLE:
- Detailed textbook-quality scientific illustration.
- White background.
- Clean, colourful but scientifically restrained palette.
- Clear outlines and distinct component boundaries.
- Good visual hierarchy.
- Accurate and recognizable structures.
- Professional educational textbook appearance.
- Clear callout lines pointing to the correct structures.
- Large labels suitable for A4 PDF printing.
- No notebook paper texture.
- No decorative borders.
- No unnecessary background objects.
- No generic flowchart or boxes-and-arrows layout.

SCIENTIFIC ACCURACY:
- Depict only the listed components and relationships.
- Do not invent additional structures.
- Do not duplicate components.
- Do not add unsupported labels.
- Keep the diagram focused on the stated subject.
- Do not replace a scientific structure with a symbolic flowchart.

LABEL REQUIREMENTS:
- Use the exact supplied label spelling.
- Keep labels short and readable.
- Do not add paragraphs or explanatory text inside the image.
- If a label cannot be placed clearly, omit it rather than
  attaching it to the wrong structure.

The image should look like a genuine educational textbook diagram,
not a presentation slide or infographic.
"""


# ============================================================
# IMAGE RESPONSE UTILITIES
# ============================================================

def decode_generated_image(image_data):
    """
    Decode and validate an image returned by Gemini.

    Gemini image response is requested as JPEG.
    The returned bytes are kept as JPEG because Pillow-based
    PDF generation can read JPEG directly.
    """

    if not image_data:
        raise ValueError(
            "The image model returned empty image data."
        )

    if isinstance(image_data, bytes):
        image_bytes = image_data

    elif isinstance(image_data, str):
        image_bytes = base64.b64decode(
            image_data,
            validate=True,
        )

    else:
        raise TypeError(
            "Unsupported image response data type."
        )

    with Image.open(BytesIO(image_bytes)) as image:
        image.verify()

    return image_bytes


# ============================================================
# GENERATE ONE CONCEPT DIAGRAM
# ============================================================

def generate_concept_diagram(
    specification,
    language="English",
):
    """
    Generate one concept-specific educational illustration.

    Requirements:
        pip install google-genai

    Requires a Google API key with access to the image model.
    """

    if genai is None:
        raise ImportError(
            "The google-genai package is missing. "
            "Install it using: pip install google-genai"
        )

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is required for image generation."
        )

    client = genai.Client(
        api_key=api_key,
    )

    prompt = generate_scientific_image_prompt(
        specification,
        language,
    )

    interaction = client.interactions.create(
        model=IMAGE_MODEL,
        input=prompt,
        response_format={
            "type": "image",
            "mime_type": "image/jpeg",
            "aspect_ratio": "4:3",
            "image_size": "1K",
        },
    )

    generated_image = getattr(
        interaction,
        "output_image",
        None,
    )

    if generated_image is None:
        raise ValueError(
            "The image model did not return an output_image."
        )

    image_data = getattr(
        generated_image,
        "data",
        None,
    )

    return decode_generated_image(image_data)


# ============================================================
# MATCH EXISTING VISUALS TO DIAGRAM SPECIFICATIONS
# ============================================================

def normalize_heading(heading):
    heading = str(heading or "").lower()

    heading = re.sub(
        r"[^a-z0-9\s]",
        " ",
        heading,
    )

    return re.sub(
        r"\s+",
        " ",
        heading,
    ).strip()


def visual_covers_heading(
    specification_heading,
    existing_visuals,
):
    target = normalize_heading(
        specification_heading
    )

    if not target:
        return False

    for visual in existing_visuals:

        existing = normalize_heading(
            visual.get("related_heading", "")
        )

        if not existing:
            continue

        if target == existing:
            return True

        if (
            len(target) >= 12
            and len(existing) >= 12
            and (
                target in existing
                or existing in target
            )
        ):
            return True

    return False


# ============================================================
# GENERATE MISSING CONCEPT DIAGRAMS
# ============================================================

def generate_missing_concept_diagrams(
    specifications,
    existing_visuals,
    language,
):
    """
    Generate illustrations for concepts not already represented
    by selected original video visuals.
    """

    if not specifications:
        return []

    if genai is None:
        st.warning(
            "Image generation is unavailable because google-genai "
            "is not installed. Only original video visuals can be used."
        )
        return []

    if not os.getenv("GOOGLE_API_KEY"):
        st.warning(
            "GOOGLE_API_KEY is missing. "
            "Missing concept illustrations cannot be generated."
        )
        return []

    generated = []

    remaining_slots = max(
        0,
        MAX_DIAGRAMS - len(existing_visuals),
    )

    generation_limit = min(
        MAX_GENERATED_DIAGRAMS,
        remaining_slots,
        len(specifications),
    )

    if generation_limit == 0:
        return []

    progress = st.progress(
        0,
        text="Generating concept-specific illustrations...",
    )

    try:
        processed = 0

        for specification in specifications:

            if len(generated) >= generation_limit:
                break

            heading = specification.get(
                "related_heading",
                "",
            )

            if visual_covers_heading(
                heading,
                existing_visuals + generated,
            ):
                continue

            processed += 1

            try:
                image_bytes = generate_concept_diagram(
                    specification,
                    language,
                )

                generated.append({
                    "number": (
                        len(existing_visuals)
                        + len(generated)
                        + 1
                    ),

                    "title": specification["title"],

                    "related_heading": (
                        specification["related_heading"]
                    ),

                    "image": image_bytes,

                    "timestamp": None,

                    "timestamp_label": (
                        "AI-generated educational illustration"
                    ),

                    "description": (
                        specification.get("purpose")
                        or
                        "AI-generated educational illustration "
                        "based on transcript-grounded study notes."
                    ),

                    "visible_labels": clean_string_list(
                        specification.get("visible_labels", []),
                        limit=12,
                    ),

                    "visual_type": specification.get(
                        "diagram_type",
                        "scientific_illustration",
                    ),

                    "confidence": None,

                    "source": (
                        "AI-generated from transcript-grounded "
                        "concept specification"
                    ),
                })

            except Exception as exc:
                st.warning(
                    f"Could not generate diagram "
                    f"'{specification.get('title', 'Untitled')}': {exc}"
                )

            progress.progress(
                min(
                    1.0,
                    len(generated) / max(1, generation_limit),
                ),
                text=(
                    f"Generated illustrations: "
                    f"{len(generated)}/{generation_limit}"
                ),
            )

    finally:
        progress.empty()

    return generated


# ============================================================
# TITLE HELPERS
# ============================================================

def extract_notes_title(notes):
    if not isinstance(notes, str):
        return "Video Study Notes"

    for line in notes.splitlines():

        match = re.match(
            r"^#{1,2}\s+(.+)$",
            line.strip(),
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
# COMPATIBILITY HELPERS
# ============================================================

def parse_diagram_material(raw_output):
    result = extract_json_from_response(
        raw_output
    )

    if isinstance(result, dict):
        diagrams = result.get("diagrams", [])

        return (
            diagrams
            if isinstance(diagrams, list)
            else []
        )

    return []


def render_diagrams(diagrams):
    """
    Compatibility function.

    Images are generated by the image-generation model,
    not by drawing generic flowcharts.
    """

    return (
        diagrams
        if isinstance(diagrams, list)
        else []
    )


# ============================================================
# COMPLETE STUDY MATERIAL GENERATION
# ============================================================

def generate_study_material(
    youtube_url,
    chat_history,
    language="English",
):
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
        raise ValueError(
            "Please provide a YouTube URL."
        )

    # --------------------------------------------------------
    # STEP 1: LOAD TRANSCRIPT
    # --------------------------------------------------------

    with st.spinner(
        "Loading the YouTube transcript..."
    ):
        documents = load_transcript(
            youtube_url
        )

    # --------------------------------------------------------
    # STEP 2: INITIALIZE GROQ
    # --------------------------------------------------------

    llm = get_llm()

    # --------------------------------------------------------
    # STEP 3: SUMMARIZE COMPLETE TRANSCRIPT
    # --------------------------------------------------------

    summaries = summarize_transcript(
        documents,
        llm,
    )

    # --------------------------------------------------------
    # STEP 4: GENERATE STUDY NOTES
    # --------------------------------------------------------

    notes = generate_notes_from_summaries(
        summaries=summaries,
        llm=llm,
        language=language,
        chat_history=chat_history,
    )

    if not notes.strip():
        raise ValueError(
            "Generated notes are empty."
        )

    # --------------------------------------------------------
    # STEP 5: EXTRACT ORIGINAL VIDEO VISUALS
    # --------------------------------------------------------

    with st.spinner(
        "Extracting original video visual references..."
    ):
        frames = extract_video_frames(
            youtube_url
        )

    # --------------------------------------------------------
    # STEP 6: ANALYSE VIDEO FRAMES
    # --------------------------------------------------------

    visual_analysis = []

    if frames:

        with st.spinner(
            f"Analysing {len(frames)} video frames..."
        ):
            visual_analysis = analyze_video_frames(
                frames=frames,
                notes=notes,
            )

    # --------------------------------------------------------
    # STEP 7: SELECT USEFUL ORIGINAL VISUALS
    # --------------------------------------------------------

    original_visuals = select_relevant_video_visuals(
        frames,
        visual_analysis,
    )

    # Respect the maximum visual count.
    original_visuals = original_visuals[:MAX_DIAGRAMS]

    # --------------------------------------------------------
    # STEP 8: IDENTIFY MISSING DIAGRAM CONCEPTS
    # --------------------------------------------------------

    specifications = []

    if len(original_visuals) < MAX_DIAGRAMS:

        with st.spinner(
            "Identifying concepts that need diagrams..."
        ):
            specifications = generate_diagram_specifications(
                llm=llm,
                notes=notes,
                language=language,
                visual_context="\n".join(
                    (
                        f"- {item.get('related_heading', '')}: "
                        f"{item.get('description', '')}"
                    )
                    for item in original_visuals
                ),
            )

    # --------------------------------------------------------
    # STEP 9: GENERATE MISSING CONCEPT ILLUSTRATIONS
    # --------------------------------------------------------

    generated_visuals = generate_missing_concept_diagrams(
        specifications=specifications,
        existing_visuals=original_visuals,
        language=language,
    )

    # --------------------------------------------------------
    # STEP 10: COMBINE VISUALS
    # --------------------------------------------------------

    selected_visuals = (
        original_visuals + generated_visuals
    )[:MAX_DIAGRAMS]

    # Update numbering after combining.
    for index, visual in enumerate(
        selected_visuals,
        start=1,
    ):
        visual["number"] = index

    # --------------------------------------------------------
    # STEP 11: GENERATE PDF
    # --------------------------------------------------------

    notes_title = extract_notes_title(
        notes
    )

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
        "complete study notes with relevant educational visuals."
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
        st.warning(
            "Enter a YouTube URL to continue."
        )
        return

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

    if st.button("Load video for questions"):

        try:
            with st.spinner(
                "Building transcript search index..."
            ):
                chain = initialize_rag_chain(
                    youtube_url
                )

            st.session_state.rag_chain = chain
            st.session_state.loaded_url = youtube_url
            st.session_state.chat_history = []
            st.session_state.generated_pdf = None
            st.session_state.generated_notes = None
            st.session_state.generated_diagrams = []

            st.success(
                "Video loaded successfully."
            )

        except Exception as exc:
            st.error(
                f"Could not load the video: {exc}"
            )

    if (
        st.session_state.rag_chain is not None
        and st.session_state.loaded_url == youtube_url
    ):

        st.subheader("Ask questions")

        for message in st.session_state.chat_history:

            with st.chat_message(message["role"]):
                st.markdown(
                    message["content"]
                )

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
                        answer = (
                            st.session_state.rag_chain.invoke(
                                question
                            )
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

    st.divider()

    st.subheader(
        "Generate complete study material"
    )

    st.write(
        "Create detailed notes, use original video visuals "
        "and generate illustrations for important missing concepts."
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

            st.session_state.generated_notes = notes
            st.session_state.generated_diagrams = diagrams
            st.session_state.generated_pdf = pdf_bytes

            st.session_state.generated_pdf_filename = (
                create_pdf_filename(
                    extract_notes_title(notes)
                )
            )

            st.success(
                f"Study material generated. "
                f"{len(diagrams)} visual(s) included."
            )

        except Exception as exc:
            st.error(
                f"Study material generation failed: {exc}"
            )

    if st.session_state.generated_notes:

        st.divider()

        st.subheader(
            "Study notes preview"
        )

        st.markdown(
            st.session_state.generated_notes
        )

        if st.session_state.generated_diagrams:

            st.subheader(
                "Educational visuals"
            )

            for diagram in st.session_state.generated_diagrams:

                st.markdown(
                    f"**{diagram.get('title', 'Educational visual')}**"
                )

                if diagram.get("timestamp_label"):
                    st.caption(
                        diagram["timestamp_label"]
                    )

                if diagram.get("description"):
                    st.caption(
                        diagram["description"]
                    )

                st.image(
                    diagram["image"],
                    use_container_width=True,
                )

                labels = diagram.get(
                    "visible_labels",
                    [],
                )

                if labels:
                    st.caption(
                        "Labels: " + ", ".join(labels)
                    )

        else:
            st.info(
                "No suitable diagrams were generated. "
                "The notes remain available without unrelated visuals."
            )

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