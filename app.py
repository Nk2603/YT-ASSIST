
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

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
# CONFIGURATION
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

YOUTUBE_URL = "https://youtu.be/p4pHsuEf4Ms?t=9"
DEFAULT_YOUTUBE_URL = YOUTUBE_URL

EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "openai/gpt-oss-120b"

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 150
RETRIEVER_K = 5


# ============================================================
# LANGUAGE MODEL
# ============================================================

def get_llm():
    """Create the configured Groq language model."""

    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    return ChatGroq(
        model=LLM_MODEL,
        temperature=0.2,
    )


# ============================================================
# YOUTUBE TRANSCRIPT
# ============================================================

def load_transcript(youtube_url):
    """Load an available English or Hindi YouTube transcript."""

    loader = YoutubeLoader.from_youtube_url(
        youtube_url,
        add_video_info=False,
        language=["en", "hi"],
    )

    documents = loader.load()

    if not documents:
        raise ValueError(
            "No supported transcript was found for this video. "
            "Check whether captions are available."
        )

    return documents


def format_docs(docs):
    """Combine retrieved documents into context text."""

    return "\n\n".join(
        document.page_content for document in docs
    )


# ============================================================
# RAG QUESTION-ANSWERING PIPELINE
# ============================================================

def initialize_rag_chain(youtube_url):
    """Build a retrieval-augmented question-answering chain."""

    documents = load_transcript(youtube_url)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "The transcript could not be split into chunks."
        )

    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError(
            "GOOGLE_API_KEY is missing. Add it to your .env file."
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
You are a helpful video-learning assistant.

Answer the question using the supplied transcript context.
If the answer is not supported by the context, say so clearly.
Do not invent facts or quotations.

Respond in the same language as the user's question.
Use simple English, Hindi, or natural Hinglish as appropriate.
Explain technical terms clearly.

Transcript context:
{context}

Question:
{question}

Answer:
"""
    )

    chain = (
        RunnableParallel(
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | get_llm()
        | StrOutputParser()
    )

    return chain


# ============================================================
# CHAT HISTORY
# ============================================================

def get_user_questions(chat_history):
    """Extract non-empty user questions from the chat history."""

    return [
        str(message.get("content", "")).strip()
        for message in chat_history
        if message.get("role") == "user"
        and str(message.get("content", "")).strip()
    ]


# ============================================================
# TRANSCRIPT SUMMARIZATION
# ============================================================

def summarize_transcript(documents, llm):
    """Summarize transcript batches to support longer videos."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=9000,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    full_text = "\n\n".join(
        document.page_content for document in documents
    )

    chunks = splitter.split_text(full_text)

    if not chunks:
        raise ValueError("The transcript is empty.")

    summaries = []

    progress = st.progress(
        0,
        text="Understanding the video transcript...",
    )

    try:
        for index, chunk in enumerate(chunks):
            response = llm.invoke(
                f"""
Create a factual, structured summary of this part of a
YouTube educational video.

Preserve:
- Important concepts and subtopics
- Definitions, explanations, and examples
- Processes, steps, and cause-effect relationships
- Technical terms, names, and important facts

Do not invent details. Do not omit important information
merely because it appears introductory.

Keep the summary concise but information-rich.
Use the transcript's language where practical.

TRANSCRIPT PART {index + 1} OF {len(chunks)}:
{chunk}
"""
            )

            content = response.content

            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"Could not summarize transcript part {index + 1}."
                )

            summaries.append(content.strip())

            progress.progress(
                (index + 1) / len(chunks),
                text=(
                    f"Processing transcript part "
                    f"{index + 1} of {len(chunks)}"
                ),
            )

    finally:
        progress.empty()

    return "\n\n".join(
        f"TRANSCRIPT SUMMARY {index + 1}:\n{summary}"
        for index, summary in enumerate(summaries)
    )


# ============================================================
# PARSE NOTES AND MERMAID DIAGRAMS
# ============================================================

def parse_study_material(raw_output):
    """
    Extract Markdown notes and Mermaid source from the model response.
    Mermaid source remains internal and is not shown to the user.
    """

    notes_match = re.search(
        r"<<<NOTES_START>>>(.*?)<<<NOTES_END>>>",
        raw_output,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if not notes_match:
        raise ValueError(
            "The model did not return notes in the expected format. "
            "Please retry generation."
        )

    notes = notes_match.group(1).strip()

    diagram_matches = re.findall(
        r"<<<DIAGRAM_(\d+)_START>>>(.*?)"
        r"<<<DIAGRAM_\1_END>>>",
        raw_output,
        flags=re.DOTALL | re.IGNORECASE,
    )

    diagrams = []

    supported_diagram_types = (
        r"^(flowchart|graph|sequenceDiagram|classDiagram|"
        r"stateDiagram(?:-v2)?|erDiagram|mindmap|timeline|"
        r"gantt|pie|journey|gitGraph|quadrantChart|"
        r"xychart-beta|block-beta)\b"
    )

    for number, source in diagram_matches:
        source = source.strip()

        # Remove optional Markdown fences.
        source = re.sub(
            r"\A```(?:mermaid)?\s*",
            "",
            source,
            flags=re.IGNORECASE,
        )

        source = re.sub(
            r"\s*```\Z",
            "",
            source,
        ).strip()

        if not re.match(
            supported_diagram_types,
            source,
            flags=re.IGNORECASE,
        ):
            continue

        diagrams.append(
            {
                "title": f"Concept Diagram {number}",
                "source": source,
            }
        )

    if not notes:
        raise ValueError(
            "The generated study notes are empty."
        )

    if not diagrams:
        raise ValueError(
            "Notes were generated, but no valid diagrams were returned. "
            "Please retry generation."
        )

    return notes, diagrams


# ============================================================
# MERMAID CLI DETECTION
# ============================================================

def get_mermaid_cli():
    """Locate and verify Mermaid CLI on Windows or other platforms."""

    mmdc_path = shutil.which("mmdc")

    if not mmdc_path:
        raise RuntimeError(
            "Python could not find Mermaid CLI. "
            "Check the PATH using 'where.exe mmdc' in PowerShell."
        )

    try:
        is_windows_script = (
            os.name == "nt"
            and Path(mmdc_path).suffix.lower() in {".cmd", ".bat"}
        )

        if is_windows_script:
            command = subprocess.list2cmdline(
                [mmdc_path, "--version"]
            )

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                shell=True,
            )
        else:
            result = subprocess.run(
                [mmdc_path, "--version"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"Mermaid CLI was found at '{mmdc_path}', "
            f"but could not be executed: {exc}"
        ) from exc

    if result.returncode != 0:
        details = (
            result.stderr or result.stdout or "No details available."
        ).strip()

        raise RuntimeError(
            f"Mermaid CLI verification failed: {details[:1500]}"
        )

    return mmdc_path


# ============================================================
# RENDER MERMAID DIAGRAMS AS PNG IMAGES
# ============================================================


def render_diagrams(diagrams):
    """
    Render Mermaid source into PNG images.

    Normalizes common formatting issues, handles Windows .CMD/.BAT
    execution, and reports useful syntax errors.
    """
    import re

    rendered = []
    mmdc_path = get_mermaid_cli()

    def clean_mermaid_source(source):
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Diagram source is empty.")

        source = source.strip().lstrip("\ufeff")

        # Remove Markdown code fences.
        source = re.sub(
            r"^\s*```(?:mermaid)?\s*",
            "",
            source,
            flags=re.IGNORECASE,
        )
        source = re.sub(r"\s*```\s*$", "", source)

        lines = [
            line.strip()
            for line in source.splitlines()
            if line.strip()
        ]

        if not lines:
            raise ValueError("Diagram source is empty after cleaning.")

        # Put a valid flowchart declaration on its own line.
        first = lines[0]
        match = re.match(
            r"^(flowchart|graph)\s+(LR|RL|TB|TD|BT)\s*(.*)$",
            first,
            flags=re.IGNORECASE,
        )

        if match:
            declaration, direction, remainder = match.groups()
            lines[0] = f"{declaration} {direction}"
            if remainder.strip():
                lines.insert(1, remainder.strip())

        # Handle a declaration accidentally joined to its first node.
        elif re.match(
            r"^(flowchart|graph)\s+(LR|RL|TB|TD|BT)\S",
            first,
            flags=re.IGNORECASE,
        ):
            match = re.match(
                r"^(flowchart|graph)\s+(LR|RL|TB|TD|BT)(.*)$",
                first,
                flags=re.IGNORECASE,
            )
            declaration, direction, remainder = match.groups()
            lines[0] = f"{declaration} {direction}"
            if remainder.strip():
                lines.insert(1, remainder.strip())

        source = "\n".join(lines)

        # Reject source without a recognizable Mermaid declaration.
        if not re.match(
            r"^(flowchart|graph|sequenceDiagram|classDiagram|"
            r"stateDiagram|erDiagram|journey|gantt|pie|mindmap|"
            r"timeline|gitGraph)\b",
            source,
            flags=re.IGNORECASE,
        ):
            raise ValueError(
                "Diagram must start with a valid Mermaid declaration, "
                "such as 'flowchart LR'."
            )

        return source

    with tempfile.TemporaryDirectory(
        prefix="yt_rag_diagrams_"
    ) as temp_dir:
        temp_path = Path(temp_dir)

        for index, diagram in enumerate(diagrams, start=1):
            source_path = temp_path / f"diagram_{index}.mmd"
            image_path = temp_path / f"diagram_{index}.png"

            try:
                source = clean_mermaid_source(diagram.get("source", ""))
            except (ValueError, AttributeError) as exc:
                raise RuntimeError(
                    f"Invalid Mermaid source for diagram {index}: {exc}"
                ) from exc

            source_path.write_text(source, encoding="utf-8")

            args = [
                mmdc_path,
                "-i", str(source_path),
                "-o", str(image_path),
                "-b", "white",
                "-t", "neutral",
            ]

            try:
                is_windows_script = (
                    os.name == "nt"
                    and Path(mmdc_path).suffix.lower() in {".cmd", ".bat"}
                )

                if is_windows_script:
                    result = subprocess.run(
                        subprocess.list2cmdline(args),
                        capture_output=True,
                        text=True,
                        timeout=120,
                        check=False,
                        shell=True,
                    )
                else:
                    result = subprocess.run(
                        args,
                        capture_output=True,
                        text=True,
                        timeout=120,
                        check=False,
                    )

            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Rendering diagram {index} timed out after 120 seconds."
                ) from exc

            except OSError as exc:
                raise RuntimeError(
                    f"Could not launch Mermaid CLI: {exc}"
                ) from exc

            if result.returncode != 0:
                error_text = (
                    result.stderr or result.stdout or "No error details returned."
                ).strip()

                # Include source to make syntax issues easier to diagnose.
                raise RuntimeError(
                    f"Mermaid failed to render diagram {index} "
                    f"(exit code {result.returncode}).\n"
                    f"Mermaid source:\n{source[:1500]}\n\n"
                    f"Details: {error_text[:2500]}"
                )

            if not image_path.is_file():
                raise RuntimeError(
                    f"Mermaid exited successfully, but diagram "
                    f"{index} was not created."
                )

            image_bytes = image_path.read_bytes()

            if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                raise RuntimeError(
                    f"Diagram {index} did not produce a valid PNG image."
                )

            rendered.append({
                "title": diagram.get(
                    "title", f"Concept Diagram {index}"
                ),
                "image": image_bytes,
            })

    if not rendered:
        raise RuntimeError("No diagrams were rendered.")

    return rendered


# ============================================================
# GENERATE COMPLETE STUDY MATERIAL
# ============================================================

def generate_study_material(
    youtube_url,
    chat_history,
    language="English",
):
    """
    Generate comprehensive study notes and diagrams,
    render the diagrams, and produce one combined PDF.
    """

    supported_languages = {"English", "Hindi", "Hinglish"}

    if language not in supported_languages:
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Choose one of {sorted(supported_languages)}."
        )

    documents = load_transcript(youtube_url)
    llm = get_llm()

    transcript_summary = summarize_transcript(
        documents,
        llm,
    )

    questions = get_user_questions(chat_history)

    if questions:
        question_text = "\n".join(
            f"- {question}"
            for question in questions[-15:]
        )
    else:
        question_text = (
            "- No questions were asked. Cover the full video."
        )

    language_instruction = {
        "English": (
            "Write the study notes in clear, simple English. "
            "Keep standard technical terms in English."
        ),
        "Hindi": (
            "Write the notes in natural Hindi using Devanagari script. "
            "Keep standard technical terms in English where clearer. "
            "Avoid unnecessary transliteration."
        ),
        "Hinglish": (
            "Write in simple Hinglish, using readable Hindi and English "
            "naturally. Preserve technical terms in English."
        ),
    }[language]

    final_prompt = f"""
You are an expert educator creating comprehensive study material
from a YouTube educational video.

LANGUAGE:
{language_instruction}

USER QUESTIONS:
{question_text}

VIDEO CONTENT SUMMARIES:
{transcript_summary}

TASK:
Create comprehensive study notes and 2-4 useful diagrams.
Produce both in this single response.

NOTES REQUIREMENTS:
- Cover important topics throughout the video.
- Explain concepts in a logical order.
- Include definitions, examples, key facts, steps, and comparisons.
- Use tables and bullet points where helpful.
- Include quick revision notes and important takeaways.
- Address the user's questions in relevant sections.
- Do not invent details absent from the supplied summaries.
- Do not include Mermaid source code in the notes.
- Use clear Markdown headings and formatting.

DIAGRAM REQUIREMENTS:
- Create 2-4 meaningful diagrams based on the video content.
- Prefer flowcharts, concept maps, process diagrams, or comparisons.
- Use valid Mermaid syntax.
- Keep labels concise and readable.
- Quote labels containing punctuation when appropriate.
- Do not create decorative or irrelevant diagrams.
- Do not include Markdown code fences around Mermaid source.
- Put no extra explanation outside the required markers.

OUTPUT FORMAT:
Return exactly this structure, without an introduction.

<<<NOTES_START>>>
Complete Markdown study notes here.
<<<NOTES_END>>>

<<<DIAGRAM_1_START>>>
flowchart TD
    A["Main concept"] --> B["Next step"]
<<<DIAGRAM_1_END>>>

<<<DIAGRAM_2_START>>>
flowchart LR
    A["Topic A"] --> B["Topic B"]
<<<DIAGRAM_2_END>>>

Continue with additional diagrams if needed, using matching numbered
start and end markers. Diagram contents must contain only Mermaid syntax.
"""

    with st.spinner(
        "Generating complete study notes and diagrams..."
    ):
        response = llm.invoke(final_prompt)

    raw_output = response.content

    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError(
            "The language model returned an empty response."
        )

    notes, diagrams = parse_study_material(raw_output)

    with st.spinner(
        f"Rendering {len(diagrams)} diagrams..."
    ):
        rendered_diagrams = render_diagrams(diagrams)

    pdf_bytes = create_study_notes_pdf(
        notes=notes,
        title="YouTube Video Study Notes",
        language=language,
        diagram_images=rendered_diagrams,
    )

    if not isinstance(pdf_bytes, bytes) or not pdf_bytes.startswith(b"%PDF"):
        raise ValueError(
            "The PDF generator did not return a valid PDF."
        )

    return notes, rendered_diagrams, pdf_bytes


# ============================================================
# STREAMLIT APPLICATION
# ============================================================

def main():
    st.set_page_config(
        page_title="YT-RAG-Assist",
        page_icon="📚",
        layout="wide",
    )

    # Initialize session state safely.
    defaults = {
        "rag_chain": None,
        "loaded_url": None,
        "chat_history": [],
        "generated_pdf": None,
        "generated_notes": None,
        "generated_diagram_count": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    st.title("📚 YT-RAG-Assist")

    st.write(
        "Ask questions about a video or generate one PDF containing "
        "comprehensive study notes and rendered diagrams."
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

    # Reset the current chat and study material when the URL changes.
    if (
        st.session_state.loaded_url is not None
        and st.session_state.loaded_url != youtube_url
    ):
        st.session_state.rag_chain = None
        st.session_state.loaded_url = None
        st.session_state.chat_history = []
        st.session_state.generated_pdf = None
        st.session_state.generated_notes = None
        st.session_state.generated_diagram_count = 0

    # --------------------------------------------------------
    # Load video for RAG questions
    # --------------------------------------------------------

    if st.button(
        "Load video for questions",
        type="secondary",
    ):
        try:
            with st.spinner(
                "Loading transcript and building search index..."
            ):
                chain = initialize_rag_chain(youtube_url)

            st.session_state.rag_chain = chain
            st.session_state.loaded_url = youtube_url
            st.session_state.chat_history = []
            st.session_state.generated_pdf = None
            st.session_state.generated_notes = None
            st.session_state.generated_diagram_count = 0

            st.success("Video loaded successfully.")

        except Exception as exc:
            st.error(f"Could not load the video: {exc}")

    # --------------------------------------------------------
    # Ask questions about the video
    # --------------------------------------------------------

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
            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

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

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as exc:
                st.error(
                    f"Could not answer the question: {exc}"
                )

    # --------------------------------------------------------
    # Generate combined study notes and diagrams PDF
    # --------------------------------------------------------

    st.divider()
    st.subheader("Generate complete study material")

    st.write(
        "Generate comprehensive notes, render diagrams automatically, "
        "and combine everything into a single PDF."
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

            st.session_state.generated_pdf = pdf_bytes
            st.session_state.generated_notes = notes
            st.session_state.generated_diagram_count = len(diagrams)

            st.success(
                f"Your PDF is ready with notes and "
                f"{len(diagrams)} rendered diagrams."
            )

        except Exception as exc:
            st.error(
                f"Study material generation failed: {exc}"
            )

    # --------------------------------------------------------
    # Download generated PDF
    # --------------------------------------------------------

    if st.session_state.generated_pdf:
        st.download_button(
            label="Download Complete Study Notes PDF",
            data=st.session_state.generated_pdf,
            file_name="youtube_study_notes.pdf",
            mime="application/pdf",
            type="primary",
        )


if __name__ == "__main__":
    main()
