
import html
import base64
import os
import re
import json
import tempfile
from io import BytesIO
from pathlib import Path
from operator import itemgetter

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.prompts import PromptTemplate

from pdf_generator import create_study_notes_pdf
from quiz_generation import generate_quiz


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
FONT_DIR = BASE_DIR / "fonts"

YOUTUBE_URL = "https://youtu.be/p4pHsuEf4Ms?t=9"
DEFAULT_YOUTUBE_URL = YOUTUBE_URL

EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "openai/gpt-oss-120b"

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 150
RETRIEVER_K = 5

SUMMARY_CHUNK_SIZE = 7500
SUMMARY_CHUNK_OVERLAP = 250
SUMMARIES_PER_NOTES_BATCH = 3
MAX_CHAT_QUESTIONS = 15

MAX_DIAGRAMS = 5
MAX_DIAGRAM_NODES = 8

MAX_VIDEO_FRAMES = 18
MAX_VIDEO_DOWNLOAD_MB = 180
MAX_FRAME_WIDTH = 1200
JPEG_QUALITY = 85
VISION_MODEL = "gemini-2.5-flash"

SUPPORTED_LANGUAGES = {
    "English",
    "Hindi",
    "Hinglish",
}


# ============================================================
# LANGUAGE MODEL
# ============================================================

def get_llm():
    """Create the Groq language model."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    return ChatGroq(
        model=LLM_MODEL,
        temperature=0.2,
        api_key=api_key,
    )


def response_to_text(response):
    """
    Safely converts different LLM response formats into plain text.
    Supports strings, dictionaries, AIMessage objects and content blocks.
    """

    if response is None:
        return ""

    # Direct string response
    if isinstance(response, str):
        return response.strip()

    # Dictionary response
    if isinstance(response, dict):
        for key in ("answer", "output", "text", "content"):
            value = response.get(key)

            if value is not None:
                return response_to_text(value)

        return str(response)

    # LangChain AIMessage or similar objects
    content = getattr(response, "content", None)

    if content is not None:
        return response_to_text(content)

    # List of content blocks
    if isinstance(response, (list, tuple)):
        parts = []

        for item in response:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                text_value = item.get("text")

                if isinstance(text_value, str):
                    parts.append(text_value)

                elif text_value is not None:
                    parts.append(response_to_text(text_value))

            else:
                item_text = getattr(item, "text", None)

                if isinstance(item_text, str):
                    parts.append(item_text)

        return "\n".join(part for part in parts if part).strip()

    return str(response)
# ============================================================
# YOUTUBE TRANSCRIPT
# ============================================================

def load_transcript(youtube_url):
    """Load the available English or Hindi YouTube transcript."""

    if not isinstance(youtube_url, str) or not youtube_url.strip():
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

    valid_documents = [
        document
        for document in documents
        if (
            hasattr(document, "page_content")
            and isinstance(document.page_content, str)
            and document.page_content.strip()
        )
    ]

    if not valid_documents:
        raise ValueError("The video transcript is empty.")

    return valid_documents


def format_docs(documents):
    """Combine retrieved LangChain documents into text."""

    if not isinstance(documents, (list, tuple)):
        return str(documents or "")

    contents = []

    for document in documents:
        page_content = getattr(document, "page_content", None)

        if isinstance(page_content, str) and page_content.strip():
            contents.append(page_content.strip())

    return "\n\n".join(contents)


def normalize_youtube_url(youtube_url):
    """Normalize common YouTube URL formats."""

    if not youtube_url:
        return ""

    url = str(youtube_url).strip()

    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)

        if match:
            return (
                "https://www.youtube.com/watch?v="
                f"{match.group(1)}"
            )

    return url


# ============================================================
# RAG QUESTION-ANSWERING PIPELINE
# ============================================================
def normalize_rag_input(value):
    """
    Normalize RAG input so the chain accepts either:
    - A plain question string
    - A dictionary containing question and chat_history
    """

    if isinstance(value, str):
        question = value.strip()
        chat_history = "No previous conversation."

    elif isinstance(value, dict):
        question = value.get("question", "")
        chat_history = value.get(
            "chat_history",
            "No previous conversation."
        )

        if not isinstance(question, str):
            raise ValueError("The question must be a string.")

        question = question.strip()

        if not isinstance(chat_history, str):
            chat_history = str(chat_history)

    else:
        raise ValueError(
            "Invalid RAG input. Expected a question string "
            "or a dictionary."
        )

    if not question:
        raise ValueError("Please enter a question.")

    return {
        "question": question,
        "chat_history": chat_history,
    }
    
def initialize_rag_chain(youtube_url):
    """Create a transcript-based FAISS retrieval chain."""

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
            "GOOGLE_API_KEY is missing. Add it to your .env file."
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=google_api_key,
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
You are an expert educational tutor and YouTube video-learning assistant.

Your job is to help the student understand concepts deeply,
not merely repeat the transcript.

CURRENT VIDEO TRANSCRIPT CONTEXT:
{context}

RECENT CONVERSATION:
{chat_history}

CURRENT STUDENT QUESTION:
{question}

==================================================
1. SOURCE ACCURACY
==================================================

- Use the retrieved transcript as the primary source for video-specific claims.
- Never claim that the video mentioned something unless supported by the transcript.
- If the transcript does not cover a requested detail, say so clearly.
- You may use reliable foundational knowledge to explain missing concepts.
- Clearly label supplementary explanations as "Additional Knowledge".
- Do not present uncertain information as established fact.
- Do not invent quotations, statistics, formulas or experimental results.

==================================================
2. FRESH EXAMPLES
==================================================

When the student requests an example, a new example, another example,
or an illustration:

- Create a fresh, original illustrative example.
- Do not simply repeat examples from the video.
- Avoid repeating examples already used in recent conversation.
- Prefer different contexts, values, scenarios or applications.
- Clearly label it "New Illustrative Example".
- Explain why the example demonstrates the concept.
- If the student asks for multiple examples, make them meaningfully different.

For numerical examples:

1. State the given values.
2. Identify the required quantity.
3. Write the relevant formula.
4. Explain the variables.
5. Substitute the values.
6. Show the calculation step by step.
7. State the final answer with units, where applicable.
8. Verify whether the result is reasonable.

Never invent source-specific numerical values.
Clearly identify hypothetical values used in illustrative examples.

==================================================
3. TEACHING APPROACH
==================================================

- Explain difficult concepts in simple language.
- Explain technical terms before using them extensively.
- Break complicated concepts into logical steps.
- Explain why each step occurs, not just what happens.
- Use practical applications and analogies when helpful.
- Connect new concepts with previously explained concepts.
- Correct misconceptions when necessary.
- Do not unnecessarily repeat the same explanation.

==================================================
4. VISUAL AND STRUCTURED REPRESENTATIONS
==================================================

Choose a representation according to the concept.

Possible representations:

- Comparison tables for differences.
- Numbered stages for sequential processes.
- Block or architecture diagrams for system components.
- Concept maps for relationships between ideas.
- Equations and worked calculations for numerical concepts.
- Pseudocode for algorithms.
- Input-process-output tables for computational systems.

Use a visual representation only when it genuinely improves understanding.

For diagrams:

- Use only meaningful and logically connected components.
- Explain the purpose of important components.
- Explain the direction of information or process flow.
- Do not invent components or relationships.
- Keep labels understandable.

When a diagram is not suitable, use a table or structured explanation instead.

==================================================
5. RESPONSE STRUCTURE
==================================================

For a detailed conceptual question, use this structure when appropriate:

1. Direct Answer
2. Explanation
3. How It Works
4. Suitable Representation
5. New Illustrative Example
6. Additional Knowledge
7. Key Takeaway

Do not force every heading into short answers.

==================================================
6. LANGUAGE
==================================================

Respond in the same language as the student's question.

Use:
- Simple English.
- Natural Hindi in Devanagari.
- Natural Hinglish.

Keep standard technical terms in English where appropriate.

Avoid unnecessarily complex vocabulary.

Now answer the student's current question accurately and helpfully.
"""
    )

    chain = (
        RunnableLambda(normalize_rag_input)
        | RunnableParallel({
            "context": (
                itemgetter("question")
                | retriever
                | format_docs
            ),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
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
    """Return valid user questions from chat history."""

    if not isinstance(chat_history, list):
        return []

    questions = []

    for message in chat_history:
        if not isinstance(message, dict):
            continue

        if message.get("role") != "user":
            continue

        content = message.get("content", "")

        if isinstance(content, str) and content.strip():
            questions.append(content.strip())

    return questions


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


def format_recent_chat_history(chat_history, max_messages=8):
    """Format recent conversation messages for contextual answers."""

    if not isinstance(chat_history, list):
        return "No previous conversation."

    recent_messages = chat_history[-max_messages:]
    formatted = []

    for message in recent_messages:
        if not isinstance(message, dict):
            continue

        role = str(message.get("role", "")).lower()
        content = message.get("content", "")

        if not isinstance(content, str):
            continue

        content = content.strip()

        if role not in {"user", "assistant"} or not content:
            continue

        content = content[:1800]

        label = "Student" if role == "user" else "Assistant"
        formatted.append(f"{label}: {content}")

    return "\n\n".join(formatted) or "No previous conversation."


def clean_chat_history(chat_history):
    """Return only valid user and assistant messages."""

    if not isinstance(chat_history, list):
        return []

    cleaned = []

    for message in chat_history:
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = message.get("content")

        if role not in {"user", "assistant"}:
            continue

        if not isinstance(content, str) or not content.strip():
            continue

        cleaned.append({
            "role": role,
            "content": content.strip(),
        })

    return cleaned


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
        raise ValueError(f"Unsupported language: {language}")

    return instructions[language]


def get_diagram_language_instruction(language):
    """Return the required language for diagram labels."""

    instructions = {
        "English": """
Write ALL diagram text in clear English.
Do not use Hindi or Hinglish.
""",

        "Hindi": """
Write ALL diagram text in natural Hindi using Devanagari script.
Standard technical terms such as API, CPU, RAM, Database,
Machine Learning and HTTP may remain in English.
Do not write Hindi using Roman letters.
""",

        "Hinglish": """
Write ALL diagram text in clear English.
Diagram labels must remain in English.
""",
    }

    return instructions.get(language, instructions["English"])


# ============================================================
# TRANSCRIPT SUMMARIZATION
# ============================================================

def summarize_transcript(documents, llm):
    """Summarize the complete transcript in manageable sections."""

    full_text = "\n\n".join(
        document.page_content.strip()
        for document in documents
        if (
            hasattr(document, "page_content")
            and isinstance(document.page_content, str)
            and document.page_content.strip()
        )
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
5. Mathematical formulas and their exact meaning.
6. Chemical equations and reactions.
7. Important facts and numerical values.
8. Processes and individual stages.
9. Cause-and-effect relationships.
10. Differences between related concepts.
11. Important examination details.

MATHEMATICAL ACCURACY:
- Preserve the original mathematical expression.
- Do not change variables or numerical values.
- Preserve fractions, powers, roots and Greek symbols.
- Use LaTeX notation where the transcript supports it.

CHEMICAL ACCURACY:
- Preserve chemical formulas and subscripts.
- Preserve reaction coefficients.
- Preserve reaction direction.
- Preserve reaction conditions when mentioned.
- Do not invent or silently balance a reaction.
- Do not change reactants or products.

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
You are an expert teacher, technical educator and academic
study-notes writer.

Create detailed, accurate and student-friendly study notes
from the supplied YouTube video summaries.

This is NOTES SECTION {batch_number} OF {total_batches}.

LANGUAGE:
{get_language_instruction(language)}

DIAGRAM LABEL LANGUAGE:
{get_diagram_language_instruction(language)}

==================================================
1. SOURCE AND KNOWLEDGE SEPARATION
==================================================

Organize information into two clearly distinguishable categories.

A. VIDEO EXPLANATION

- Include concepts, definitions, facts and examples supported
  by the supplied video summaries.
- Preserve the original meaning.
- Do not attribute outside information to the video.

B. ADDITIONAL KNOWLEDGE

- Add useful foundational explanations where they improve
  understanding of a topic.
- Explain important prerequisites and missing conceptual links.
- Add relevant applications and practical context.
- Clearly mark this content using:

  **Additional Knowledge:**

- Do not claim that this information was explained in the video.
- Do not add unrelated information merely to increase note length.
- Avoid unsupported statistics and uncertain claims.

==================================================
2. EXAMPLES
==================================================

Include relevant examples that make difficult concepts easier.

Use these labels:

**Video Example:**
For an example actually supported by the video.

**New Illustrative Example:**
For a newly created example that is not claimed to be from the video.

For new examples:

- Use realistic and understandable situations.
- Prefer examples different from the video's examples.
- Explain the concept demonstrated by the example.
- Explain the important steps.
- Avoid repeating the same example in multiple sections.

For numerical examples, include:

1. Given values.
2. Required quantity.
3. Formula.
4. Meaning of variables.
5. Substitution.
6. Step-by-step calculation.
7. Final answer and units.
8. Brief interpretation.

Clearly label hypothetical values.

Do not invent experimental results or source-specific numerical facts.

==================================================
3. CONCEPTUAL EXPLANATIONS
==================================================

For every important topic:

- Explain what it is.
- Explain why it is needed.
- Explain how it works.
- Describe its important components.
- Explain relationships between components.
- Include advantages, limitations or applications
  where relevant.
- Explain important technical terms.
- Include comparisons where they improve understanding.

Avoid unnecessarily long explanations of simple topics.

==================================================
4. VISUAL REPRESENTATIONS
==================================================

Use a suitable representation when it genuinely improves learning.

Possible representations:

- Architecture diagrams.
- Block diagrams.
- Process diagrams.
- Concept maps.
- Comparison tables.
- Input-process-output representations.
- Numbered workflows.
- Mathematical derivations.
- Worked numerical examples.

Do not force a diagram into every topic.

For a complex process, explain its stages in the correct order.

For an architecture, explain the role of each component
and how the components interact.

For comparisons, prefer a clear Markdown table.

Do not create ASCII diagrams or Mermaid flowcharts in the notes.

Do not invent components, relationships or process stages.

==================================================
5. FORMATTING
==================================================

- Use Markdown headings.
- Use ## for main topics.
- Use ### for subtopics.
- Use short, readable paragraphs.
- Use bullet points where helpful.
- Use bold for important terms.
- Use Markdown tables for comparisons.
- Do not create a repeated cover title.
- Avoid unnecessary repetition between sections.

==================================================
6. MATHEMATICAL FORMULAS
==================================================

- Use proper LaTeX notation.
- Put important standalone formulas inside display math delimiters.
- Use \\[ ... \\] for display equations.
- Use \\( ... \\) for inline mathematical expressions.
- Use \\frac{{a}}{{b}} for fractions.
- Use ^ for superscripts.
- Use _ for subscripts.
- Preserve roots, Greek letters and mathematical operators.
- Explain every important variable.
- Preserve the original mathematical meaning.
- Never fabricate a formula.

==================================================
7. CHEMICAL REACTIONS
==================================================

- Preserve chemical formulas and subscripts.
- Preserve coefficients and reaction arrows.
- Preserve reaction conditions when mentioned.
- Use proper chemical notation.
- Put important reactions in separate reaction blocks.
- Use the [REACTION] marker before a reaction block.
- Explain reactants, products and conditions.
- Never invent a reaction.
- Never silently change an equation.

==================================================
8. EXAMINATION PREPARATION
==================================================

Include where relevant:

- Important definitions.
- Key facts.
- Differences between related concepts.
- Important formulas.
- Applications.
- Common misconceptions.
- Important points that students should remember.

Do not turn every topic into a list of examination questions.

==================================================
9. STUDENT QUESTIONS
==================================================

Use these questions to identify topics that deserve additional
explanation or examples:

{questions_context}

==================================================
10. VIDEO SUMMARIES
==================================================

{summary_text}

==================================================
FINAL QUALITY CHECK
==================================================

Before returning the notes, check:

- Are video-supported facts separated from supplementary knowledge?
- Are newly created examples clearly labelled?
- Are important concepts explained rather than merely named?
- Are formulas and reactions preserved accurately?
- Are numerical examples solved step by step?
- Are tables and diagrams appropriate for their topics?
- Is unnecessary repetition avoided?

Generate only the Markdown notes for this section.
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
    language="English",
    chat_history=None,
):
    """Generate complete study notes from transcript summaries."""

    if not summaries:
        raise ValueError("No transcript summaries were provided.")

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")

    chat_history = clean_chat_history(chat_history or [])
    questions_context = build_questions_context(chat_history)

    total_batches = (
        len(summaries) + SUMMARIES_PER_NOTES_BATCH - 1
    ) // SUMMARIES_PER_NOTES_BATCH

    generated_sections = []

    progress = st.progress(
        0,
        text="Preparing your study notes...",
    )

    try:
        for batch_index in range(total_batches):

            start = batch_index * SUMMARIES_PER_NOTES_BATCH
            end = start + SUMMARIES_PER_NOTES_BATCH

            summary_batch = summaries[start:end]

            section = generate_notes_batch(
                llm=llm,
                summary_batch=summary_batch,
                batch_number=batch_index + 1,
                total_batches=total_batches,
                language=language,
                questions_context=questions_context,
            )

            if not section.strip():
                raise ValueError(
                    f"Notes section {batch_index + 1} is empty."
                )

            generated_sections.append(section.strip())

            progress.progress(
                (batch_index + 1) / total_batches,
                text=(
                    f"Generating notes: "
                    f"{batch_index + 1}/{total_batches} sections"
                ),
            )

    finally:
        progress.empty()

    complete_notes = "\n\n".join(generated_sections)

    if not complete_notes.strip():
        raise ValueError("Study notes generation returned empty content.")

    return complete_notes


# ============================================================
# NOTES HEADING EXTRACTION
# ============================================================

def extract_note_headings(notes):
    """Extract Markdown headings from generated notes."""

    if not isinstance(notes, str):
        return []

    headings = []

    for line in notes.splitlines():

        match = re.match(
            r"^\s{0,3}#{1,4}\s+(.+)$",
            line,
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
# DIAGRAM JSON PARSER
# ============================================================

def extract_json_from_response(response_text):
    """Extract JSON from model output, including fenced responses."""

    if not isinstance(response_text, str) or not response_text.strip():
        return None

    cleaned = response_text.strip()

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
    )

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()

    for index, character in enumerate(cleaned):
        if character not in "{[":
            continue

        try:
            result, _ = decoder.raw_decode(cleaned[index:])
            return result
        except json.JSONDecodeError:
            continue

    return None


# ============================================================
# DIAGRAM SPECIFICATION GENERATION
# ============================================================

def generate_diagram_specifications(
    llm,
    notes,
    language,
    visual_context="",
):
    """Generate simple, topic-specific diagram specifications."""

    headings = extract_note_headings(notes)

    headings_text = "\n".join(
        f"- {heading}"
        for heading in headings[:40]
    )

    prompt = f"""
You are an educational diagram designer.

Create a small number of useful diagrams for the study notes.

LANGUAGE:
{get_language_instruction(language)}

DIAGRAM LABEL LANGUAGE:
{get_diagram_language_instruction(language)}

NOTES:
{notes[:18000]}

MAIN TOPICS:
{headings_text}

DIAGRAM REQUIREMENTS:

- Generate at most {MAX_DIAGRAMS} diagrams.
- Generate diagrams only when they genuinely improve understanding.
- Select the representation appropriate to the topic.

Representation selection:

1. PROCESS:
   Use for sequential operations, workflows and procedures.

2. ARCHITECTURE:
   Use for systems containing multiple interacting components.

3. BLOCK:
   Use for input-process-output systems and functional components.

4. CONCEPT:
   Use for relationships between concepts and their subtopics.

- Use only information supported by the supplied notes.
- Supplementary knowledge may be represented only if it is
  explicitly included in the notes.
- Do not invent components or connections.
- Preserve correct process order.
- Show meaningful relationships.
- Keep node labels short and readable.
- Use a maximum of {MAX_DIAGRAM_NODES} nodes per diagram.
- Match each diagram to a specific notes heading.
- Do not generate decorative or unrelated diagrams.
- Do not force a diagram for a simple definition.
- Prefer a comparison table or written explanation when a diagram
  would not improve understanding.

Return valid JSON only.

Required structure:

{{
  "diagrams": [
    {{
      "title": "Diagram title",
      "related_heading": "Exact or closely matching notes heading",
      "description": "One sentence explaining the diagram",
      "diagram_type": "process",
      "direction": "vertical",
      "nodes": [
        {{
          "id": "A",
          "label": "First step",
          "details": "Short explanation"
        }},
        {{
          "id": "B",
          "label": "Second step",
          "details": "Short explanation"
        }}
      ],
      "connections": [
        {{
          "from": "A",
          "to": "B",
          "label": ""
        }}
      ]
    }}
  ]
}}

Allowed diagram_type values:
- process
- architecture
- block
- concept

Allowed direction values:
- vertical
- horizontal

If the notes do not need diagrams, return:
{{"diagrams": []}}

Return JSON only. Do not use Markdown code fences.
"""

    try:
        response = llm.invoke(prompt)

        result = extract_json_from_response(
            response_to_text(response)
        )

        if isinstance(result, list):
            diagrams = result

        elif isinstance(result, dict):
            diagrams = result.get("diagrams", [])

        else:
            return []

        if not isinstance(diagrams, list):
            return []

        validated = []

        for diagram in diagrams:

            if not isinstance(diagram, dict):
                continue

            nodes = diagram.get("nodes", [])
            connections = diagram.get("connections", [])

            if not isinstance(nodes, list):
                continue

            if not isinstance(connections, list):
                connections = []

            nodes = [
                node
                for node in nodes[:MAX_DIAGRAM_NODES]
                if (
                    isinstance(node, dict)
                    and str(node.get("id", "")).strip()
                    and str(node.get("label", "")).strip()
                )
            ]

            if len(nodes) < 2:
                continue

            valid_ids = {
                str(node["id"]).strip()
                for node in nodes
            }

            valid_connections = []

            for connection in connections:

                if not isinstance(connection, dict):
                    continue

                source = str(
                    connection.get("from", "")
                ).strip()

                target = str(
                    connection.get("to", "")
                ).strip()

                if (
                    source in valid_ids
                    and target in valid_ids
                    and source != target
                ):
                    valid_connections.append({
                        "from": source,
                        "to": target,
                        "label": str(
                            connection.get("label", "")
                        )[:60],
                    })

            if not valid_connections:
                continue

            diagram_type = str(
                diagram.get("diagram_type", "block")
            ).lower()

            if diagram_type not in {
                "process",
                "architecture",
                "block",
                "concept",
            }:
                diagram_type = "block"

            direction = str(
                diagram.get("direction", "vertical")
            ).lower()

            if direction not in {"vertical", "horizontal"}:
                direction = "vertical"

            validated.append({
                "title": str(
                    diagram.get("title", "Concept Diagram")
                )[:150],
                "related_heading": str(
                    diagram.get("related_heading", "")
                )[:150],
                "description": str(
                    diagram.get("description", "")
                )[:500],
                "diagram_type": diagram_type,
                "visual_type": diagram_type,
                "direction": direction,
                "nodes": nodes,
                "connections": valid_connections,
                "source": "AI-generated educational diagram",
            })

            if len(validated) >= MAX_DIAGRAMS:
                break

        return validated

    except Exception as exc:
        st.warning(
            "Diagram specification generation failed. "
            f"Notes will still be generated. Details: {exc}"
        )
        return []


# ============================================================
# BROWSER-BASED DIAGRAM RENDERING
# ============================================================

def get_diagram_font_css():
    """Embed available Noto fonts directly into the HTML."""

    font_definitions = [
        ("NotoSans", "NotoSans-Regular.ttf", "400"),
        ("NotoSans", "NotoSans-Bold.ttf", "700"),
        ("NotoDevanagari", "NotoSansDevanagari-Regular.ttf", "400"),
        ("NotoDevanagari", "NotoSansDevanagari-Bold.ttf", "700"),
    ]

    css = []

    for family, filename, weight in font_definitions:
        font_path = FONT_DIR / filename

        if not font_path.exists():
            continue

        try:
            encoded_font = base64.b64encode(
                font_path.read_bytes()
            ).decode("ascii")

            css.append(
                f"""
                @font-face {{
                    font-family: '{family}';
                    src: url(data:font/ttf;base64,{encoded_font})
                         format('truetype');
                    font-weight: {weight};
                    font-style: normal;
                }}
                """
            )

        except Exception as error:
            print(f"Could not load font {filename}: {error}")

    return "\n".join(css)


def _safe_diagram_text(value):
    """Safely convert model-generated values into HTML text."""

    if value is None:
        return ""

    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)

    return html.escape(str(value))


def build_diagram_html(specification):
    """Build a self-contained HTML diagram."""

    if not isinstance(specification, dict):
        return None

    title = _safe_diagram_text(
        specification.get("title", "Concept Diagram")
    )

    description = _safe_diagram_text(
        specification.get("description", "")
    )

    diagram_type = str(
        specification.get("diagram_type", "process")
    ).lower()

    direction = str(
        specification.get("direction", "vertical")
    ).lower()

    nodes = specification.get("nodes", [])
    connections = specification.get("connections", [])

    if not isinstance(nodes, list):
        nodes = []

    if not isinstance(connections, list):
        connections = []

    nodes = nodes[:MAX_DIAGRAM_NODES]

    if not nodes:
        return None

    node_ids = set()
    cleaned_nodes = []

    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue

        node_id = str(node.get("id", f"node_{index + 1}"))
        node_id = re.sub(r"[^a-zA-Z0-9_-]", "_", node_id)

        if not node_id:
            node_id = f"node_{index + 1}"

        if node_id in node_ids:
            node_id = f"{node_id}_{index + 1}"

        node_ids.add(node_id)

        cleaned_nodes.append({
            "id": node_id,
            "label": _safe_diagram_text(
                node.get("label", f"Step {index + 1}")
            ),
            "details": _safe_diagram_text(
                node.get("details", "")
            ),
        })

    nodes = cleaned_nodes

    if not nodes:
        return None

    if direction == "horizontal" and diagram_type == "process":
        columns = min(len(nodes), 4)
    elif diagram_type in ("architecture", "concept"):
        columns = 2 if len(nodes) > 1 else 1
    else:
        columns = 1

    columns = max(1, columns)

    valid_connections = []

    for connection in connections:
        if not isinstance(connection, dict):
            continue

        source = str(connection.get("from", ""))
        target = str(connection.get("to", ""))

        if source in node_ids and target in node_ids:
            valid_connections.append({
                "from": source,
                "to": target,
                "label": _safe_diagram_text(
                    connection.get("label", "")
                ),
            })

    node_cards = []

    for node in nodes:
        details_html = ""

        if node["details"]:
            details_html = (
                f'<div class="node-details">{node["details"]}</div>'
            )

        node_cards.append(
            f"""
            <div class="node-card" data-node-id="{node["id"]}">
                <div class="node-label">{node["label"]}</div>
                {details_html}
            </div>
            """
        )

    connections_json = json.dumps(
        valid_connections,
        ensure_ascii=False,
    )

    font_css = get_diagram_font_css()

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">

        <style>
            {font_css}

            * {{
                box-sizing: border-box;
            }}

            body {{
                margin: 0;
                padding: 0;
                background: #ffffff;
            }}

            #diagram {{
                width: 1100px;
                padding: 42px;
                background: #ffffff;
                color: #172033;
                font-family:
                    'NotoDevanagari',
                    'NotoSans',
                    'Nirmala UI',
                    'Mangal',
                    sans-serif;
            }}

            .diagram-title {{
                font-size: 31px;
                font-weight: 700;
                text-align: center;
                line-height: 1.45;
                color: #25345c;
                margin-bottom: 12px;
            }}

            .diagram-description {{
                max-width: 900px;
                margin: 0 auto 34px auto;
                text-align: center;
                font-size: 19px;
                line-height: 1.65;
                color: #586174;
            }}

            .diagram-canvas {{
                position: relative;
                width: 100%;
                padding: 20px 10px;
            }}

            .nodes-grid {{
                position: relative;
                z-index: 2;
                display: grid;
                grid-template-columns:
                    repeat({columns}, minmax(0, 1fr));
                gap: 86px 90px;
                align-items: stretch;
            }}

            .node-card {{
                min-height: 125px;
                padding: 25px 22px;
                border: 2px solid #8c9de0;
                border-radius: 17px;
                background: #f2f5ff;
                box-shadow: 0 5px 13px rgba(35, 50, 90, 0.10);
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                text-align: center;
                overflow-wrap: anywhere;
                word-break: normal;
            }}

            .node-card:nth-child(3n + 2) {{
                background: #eef9f6;
                border-color: #69ad9b;
            }}

            .node-card:nth-child(3n) {{
                background: #fff6e9;
                border-color: #d9aa64;
            }}

            .node-label {{
                font-size: 23px;
                line-height: 1.5;
                font-weight: 700;
                color: #25345c;
            }}

            .node-details {{
                margin-top: 11px;
                font-size: 17px;
                line-height: 1.55;
                color: #48536a;
                white-space: pre-wrap;
            }}

            .connections-layer {{
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                z-index: 1;
                overflow: visible;
                pointer-events: none;
            }}

            .connection-label {{
                font-family:
                    'NotoDevanagari',
                    'NotoSans',
                    'Nirmala UI',
                    'Mangal',
                    sans-serif;
                font-size: 15px;
                fill: #34415d;
                text-anchor: middle;
                dominant-baseline: middle;
            }}
        </style>
    </head>

    <body>
        <div id="diagram">
            <div class="diagram-title">{title}</div>

            {
                f'<div class="diagram-description">{description}</div>'
                if description else ''
            }

            <div class="diagram-canvas" id="diagram-canvas">

                <svg
                    class="connections-layer"
                    id="connections-layer"
                    aria-hidden="true"
                >
                    <defs>
                        <marker
                            id="arrowhead"
                            markerWidth="10"
                            markerHeight="10"
                            refX="8"
                            refY="3"
                            orient="auto"
                            markerUnits="strokeWidth"
                        >
                            <path
                                d="M0,0 L0,6 L9,3 z"
                                fill="#6676ad"
                            />
                        </marker>
                    </defs>
                </svg>

                <div class="nodes-grid">
                    {"".join(node_cards)}
                </div>

            </div>
        </div>

        <script>
            const connections = {connections_json};

            function drawConnections() {{
                const canvas = document.getElementById("diagram-canvas");
                const svg = document.getElementById("connections-layer");

                const canvasRect = canvas.getBoundingClientRect();

                svg.setAttribute(
                    "viewBox",
                    `0 0 ${{canvasRect.width}} ${{canvasRect.height}}`
                );

                svg.setAttribute("width", canvasRect.width);
                svg.setAttribute("height", canvasRect.height);

                svg.innerHTML = `
                    <defs>
                        <marker
                            id="arrowhead"
                            markerWidth="10"
                            markerHeight="10"
                            refX="8"
                            refY="3"
                            orient="auto"
                            markerUnits="strokeWidth"
                        >
                            <path
                                d="M0,0 L0,6 L9,3 z"
                                fill="#6676ad"
                            />
                        </marker>
                    </defs>
                `;

                connections.forEach(connection => {{
                    const source = document.querySelector(
                        `[data-node-id="${{connection.from}}"]`
                    );

                    const target = document.querySelector(
                        `[data-node-id="${{connection.to}}"]`
                    );

                    if (!source || !target) return;

                    const sourceRect = source.getBoundingClientRect();
                    const targetRect = target.getBoundingClientRect();

                    const x1 =
                        sourceRect.left + sourceRect.width / 2
                        - canvasRect.left;

                    const y1 =
                        sourceRect.top + sourceRect.height / 2
                        - canvasRect.top;

                    const x2 =
                        targetRect.left + targetRect.width / 2
                        - canvasRect.left;

                    const y2 =
                        targetRect.top + targetRect.height / 2
                        - canvasRect.top;

                    const line = document.createElementNS(
                        "http://www.w3.org/2000/svg",
                        "line"
                    );

                    line.setAttribute("x1", x1);
                    line.setAttribute("y1", y1);
                    line.setAttribute("x2", x2);
                    line.setAttribute("y2", y2);
                    line.setAttribute("stroke", "#6676ad");
                    line.setAttribute("stroke-width", "3");
                    line.setAttribute(
                        "marker-end",
                        "url(#arrowhead)"
                    );

                    svg.appendChild(line);

                    if (connection.label) {{
                        const text = document.createElementNS(
                            "http://www.w3.org/2000/svg",
                            "text"
                        );

                        text.setAttribute(
                            "x",
                            (x1 + x2) / 2
                        );

                        text.setAttribute(
                            "y",
                            (y1 + y2) / 2 - 9
                        );

                        text.setAttribute(
                            "class",
                            "connection-label"
                        );

                        text.textContent = connection.label;
                        svg.appendChild(text);
                    }}
                }});
            }}

            window.addEventListener("load", async () => {{
                await document.fonts.ready;

                requestAnimationFrame(() => {{
                    drawConnections();
                }});
            }});
        </script>
    </body>
    </html>
    """


def render_single_diagram(specification, page=None):
    """Render one diagram as PNG bytes using Chromium."""

    from playwright.sync_api import sync_playwright

    html_document = build_diagram_html(specification)

    if not html_document:
        return None

    if page is not None:
        page.set_content(
            html_document,
            wait_until="load",
            timeout=30000,
        )

        page.evaluate("""
            async () => {
                await document.fonts.ready;
                await new Promise(resolve =>
                    requestAnimationFrame(resolve)
                );
            }
        """)

        return page.locator("#diagram").screenshot(
            type="png",
            animations="disabled",
        )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()

        try:
            standalone_page = browser.new_page(
                viewport={"width": 1200, "height": 1000},
                device_scale_factor=2,
            )

            standalone_page.set_content(
                html_document,
                wait_until="load",
            )

            standalone_page.evaluate("""
                async () => {
                    await document.fonts.ready;
                    await new Promise(resolve =>
                        requestAnimationFrame(resolve)
                    );
                }
            """)

            return standalone_page.locator("#diagram").screenshot(
                type="png",
                animations="disabled",
            )

        finally:
            browser.close()


def render_diagrams(diagrams):
    """Render diagram specifications using Playwright Chromium."""

    if not diagrams:
        return []

    rendered_diagrams = []

    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()

            try:
                page = browser.new_page(
                    viewport={"width": 1200, "height": 1000},
                    device_scale_factor=2,
                )

                for index, specification in enumerate(
                    diagrams[:MAX_DIAGRAMS],
                    start=1,
                ):
                    try:
                        image_bytes = render_single_diagram(
                            specification,
                            page=page,
                        )

                        if not image_bytes:
                            continue

                        rendered_diagrams.append({
                            "number": index,
                            "title": specification.get(
                                "title",
                                f"Diagram {index}",
                            ),
                            "related_heading": specification.get(
                                "related_heading",
                                "",
                            ),
                            "description": specification.get(
                                "description",
                                "",
                            ),
                            "diagram_type": specification.get(
                                "diagram_type",
                                "concept",
                            ),
                            "visual_type": specification.get(
                                "diagram_type",
                                "concept",
                            ),
                            "nodes": specification.get("nodes", []),
                            "connections": specification.get(
                                "connections",
                                [],
                            ),
                            "image": image_bytes,
                            "source": "AI-generated concept illustration",
                        })

                    except Exception as error:
                        print(
                            f"Diagram {index} rendering failed: {error}"
                        )

            finally:
                browser.close()

    except Exception as error:
        st.warning(
            "Diagram rendering could not be completed. "
            "Please check that Playwright Chromium is installed. "
            f"Details: {error}"
        )

    return rendered_diagrams


def parse_diagram_material(raw_output):
    """Compatibility helper for older code."""

    result = extract_json_from_response(raw_output)

    if isinstance(result, list):
        return result

    if isinstance(result, dict):
        diagrams = result.get("diagrams", [])
        return diagrams if isinstance(diagrams, list) else []

    return []


# ============================================================
# PREMIUM VIDEO FRAME ANALYSIS
# ============================================================

def extract_video_frames_premium(youtube_url):
    """Extract original video frames for future Premium Notes."""

    try:
        import cv2
        import yt_dlp

    except ImportError as exc:
        raise ImportError(
            "Premium frame extraction requires "
            "yt-dlp and opencv-python-headless."
        ) from exc

    frames = []

    with tempfile.TemporaryDirectory(
        prefix="yt_premium_visuals_"
    ) as temp_dir:

        video_path = os.path.join(
            temp_dir,
            "premium_video.mp4",
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
                    Path(temp_dir).glob("premium_video.*")
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

            sample_count = min(
                MAX_VIDEO_FRAMES,
                max(1, int(duration // 60)),
            )

            if sample_count == 1:
                timestamps = [duration * 0.5]

            else:
                timestamps = [
                    duration * (index + 1)
                    / (sample_count + 1)
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

        finally:
            if capture is not None:
                capture.release()

    return frames


def analyze_video_frames_premium(frames, notes):
    """Analyse original video frames with Gemini."""

    if not frames:
        return []

    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError(
            "GOOGLE_API_KEY is required for Premium visual analysis."
        )

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage

    except ImportError as exc:
        raise ImportError(
            "Install langchain-google-genai "
            "to enable Premium visual analysis."
        ) from exc

    vision_llm = ChatGoogleGenerativeAI(
        model=VISION_MODEL,
        temperature=0.1,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    headings = extract_note_headings(notes)

    prompt = f"""
You are analysing original video frames for an educational PDF.

Relevant study topics:
{chr(10).join(headings[:50])}

Select only genuinely useful educational visuals.

Reject:
- Talking-head shots.
- Random backgrounds.
- Unrelated images.
- Unclear visuals.
- Subtitle-only frames.

Do not invent labels or structures.
Analyse only what is actually visible.

Return valid JSON only:

{{
  "frames": [
    {{
      "frame_number": 1,
      "is_useful": true,
      "confidence": 0.9,
      "related_heading": "Topic heading",
      "description": "Description of the visible visual",
      "visible_labels": []
    }}
  ]
}}

If nothing is useful, return:
{{"frames": []}}
"""

    content = [{
        "type": "text",
        "text": prompt,
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

    response = vision_llm.invoke([
        HumanMessage(content=content)
    ])

    result = extract_json_from_response(
        response_to_text(response)
    )

    if isinstance(result, dict):
        frame_results = result.get("frames", [])
        return frame_results if isinstance(frame_results, list) else []

    return []


def timestamp_to_text(seconds):
    """Convert seconds into HH:MM:SS."""

    seconds = max(0, int(seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{remaining_seconds:02d}"
    )


def select_relevant_video_visuals_premium(
    frames,
    analysis,
):
    """Select useful original frames for future Premium Notes."""

    if not frames or not analysis:
        return []

    selected = []
    used_numbers = set()

    for item in analysis:

        if not isinstance(item, dict):
            continue

        if item.get("is_useful") is not True:
            continue

        try:
            confidence = float(item.get("confidence", 0))
            frame_number = int(item.get("frame_number", 0))
        except (TypeError, ValueError):
            continue

        if confidence < 0.65:
            continue

        if (
            frame_number < 1
            or frame_number > len(frames)
            or frame_number in used_numbers
        ):
            continue

        frame = frames[frame_number - 1]

        try:
            with Image.open(BytesIO(frame["image"])) as image:
                output = BytesIO()

                image.convert("RGB").save(
                    output,
                    format="PNG",
                    optimize=True,
                )

                png_bytes = output.getvalue()

        except Exception:
            continue

        title = str(
            item.get("related_heading", "Video Visual")
        ).strip()

        visible_labels = item.get("visible_labels", [])

        if not isinstance(visible_labels, list):
            visible_labels = []

        selected.append({
            "number": len(selected) + 1,
            "title": title,
            "related_heading": title,
            "description": str(
                item.get("description", "")
            ),
            "timestamp": frame["timestamp"],
            "timestamp_label": timestamp_to_text(
                frame["timestamp"]
            ),
            "image": png_bytes,
            "visible_labels": visible_labels,
            "source": "Original YouTube video frame",
        })

        used_numbers.add(frame_number)

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
# GENERATE COMPLETE STUDY MATERIAL
# ============================================================

def generate_study_material(
    youtube_url,
    chat_history,
    language="English",
):
    """
    Generate Standard Notes and a notebook-style PDF.

    Workflow:
    1. Load transcript.
    2. Summarize transcript.
    3. Generate detailed study notes.
    4. Generate diagram specifications using Groq.
    5. Render diagrams using Playwright Chromium.
    6. Generate the PDF.

    Returns:
        notes, rendered_diagrams, pdf_bytes
    """

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")

    if not isinstance(youtube_url, str) or not youtube_url.strip():
        raise ValueError("Please provide a YouTube URL.")

    chat_history = clean_chat_history(chat_history or [])

    with st.spinner("Loading the YouTube transcript..."):
        documents = load_transcript(youtube_url)

    llm = get_llm()

    summaries = summarize_transcript(
        documents,
        llm,
    )

    notes = generate_notes_from_summaries(
        summaries=summaries,
        llm=llm,
        language=language,
        chat_history=chat_history,
    )

    if not notes.strip():
        raise ValueError("The generated study notes are empty.")

    with st.spinner("Designing topic-specific educational diagrams..."):
        diagram_specs = generate_diagram_specifications(
            llm=llm,
            notes=notes,
            language=language,
        )

    with st.spinner("Rendering educational diagrams in Chromium..."):
        rendered_diagrams = render_diagrams(diagram_specs)

    notes_title = extract_notes_title(notes)

    with st.spinner("Preparing your notebook-style PDF..."):
        pdf_bytes = create_study_notes_pdf(
            notes=notes,
            title=notes_title,
            language=language,
            diagram_images=rendered_diagrams,
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
        rendered_diagrams,
        pdf_bytes,
    )


# ============================================================
# QUIZ GENERATION INTEGRATION
# ============================================================

def generate_video_quiz(
    youtube_url,
    notes="",
    chat_history=None,
    language="English",
    question_count=10,
    difficulty="Mixed",
    question_types=None,
    previous_questions=None,
):
    """
    Generate a quiz using:
    - Complete YouTube transcript
    - Generated study notes
    - Learner chat history
    - Previously generated questions
    """

    if not isinstance(youtube_url, str) or not youtube_url.strip():
        raise ValueError("Please provide a valid YouTube URL.")

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            "Invalid quiz language."
        )

    try:
        question_count = int(question_count)
    except (TypeError, ValueError):
        raise ValueError(
            "Question count must be a valid integer."
        )

    if not 1 <= question_count <= 30:
        raise ValueError(
            "Question count must be between 1 and 30."
        )

    difficulty = str(difficulty or "Mixed").strip()

    if difficulty not in {"Easy", "Medium", "Hard", "Mixed"}:
        raise ValueError("Invalid quiz difficulty.")

    cleaned_previous_questions = []

    if isinstance(previous_questions, (list, tuple)):
        for item in previous_questions:

            if isinstance(item, dict):
                question_text = item.get("question", "")

            else:
                question_text = item

            if isinstance(question_text, str) and question_text.strip():
                cleaned_previous_questions.append(
                    question_text.strip()[:1000]
                )

    cleaned_previous_questions = (
        cleaned_previous_questions[-150:]
    )

    documents = load_transcript(youtube_url.strip())
    transcript = format_docs(documents)

    if not transcript.strip():
        raise ValueError("The YouTube transcript is empty.")

    llm = get_llm()

    quiz = generate_quiz(
        llm=llm,
        transcript=transcript,
        notes=notes or "",
        chat_history=clean_chat_history(chat_history or []),
        language=language,
        question_count=question_count,
        difficulty=difficulty,
        question_types=question_types,
        previous_questions=cleaned_previous_questions,
    )

    if not isinstance(quiz, dict):
        raise ValueError(
            "Quiz generation returned an invalid response."
        )

    questions = quiz.get("questions")

    if not isinstance(questions, list):
        raise ValueError(
            "Quiz generation did not return a valid question list."
        )

    if len(questions) != question_count:
        raise ValueError(
            f"Expected {question_count} questions, "
            f"but received {len(questions)}."
        )

    quiz["source_url"] = normalize_youtube_url(youtube_url)

    return quiz


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

    # Ensure the session history always has a safe structure.
    st.session_state.chat_history = clean_chat_history(
        st.session_state.chat_history
    )

    st.title("📚 YT-RAG-Assist")

    st.write(
        "Ask questions about a YouTube video and generate "
        "complete study notes with topic-specific diagrams."
    )

    youtube_url = st.text_input(
        "YouTube video URL",
        value=DEFAULT_YOUTUBE_URL,
        key="youtube_url_input",
    ).strip()

    language = st.selectbox(
        "Study notes language",
        ["English", "Hindi", "Hinglish"],
        key="study_language",
    )

    if not youtube_url:
        st.warning("Enter a YouTube URL to continue.")
        return

    # --------------------------------------------------------
    # RESET WHEN VIDEO URL CHANGES
    # --------------------------------------------------------

    current_url = normalize_youtube_url(youtube_url)
    loaded_url = normalize_youtube_url(
        st.session_state.loaded_url or ""
    )

    if loaded_url and loaded_url != current_url:

        st.session_state.rag_chain = None
        st.session_state.loaded_url = None
        st.session_state.chat_history = []
        st.session_state.generated_pdf = None
        st.session_state.generated_notes = None
        st.session_state.generated_diagrams = []

    # --------------------------------------------------------
    # LOAD VIDEO FOR RAG
    # --------------------------------------------------------

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
            st.error(f"Could not load the video: {exc}")

    # --------------------------------------------------------
    # CHAT INTERFACE
    # --------------------------------------------------------

    video_is_loaded = (
        st.session_state.rag_chain is not None
        and normalize_youtube_url(
            st.session_state.loaded_url or ""
        ) == current_url
    )

    if video_is_loaded:

        st.subheader("Ask questions")

        # Display previous conversation safely.
        for message in st.session_state.chat_history:

            if not isinstance(message, dict):
                continue

            role = message.get("role")
            content = message.get("content", "")

            if role not in {"user", "assistant"}:
                continue

            if not isinstance(content, str):
                continue

            with st.chat_message(role):
                st.markdown(content)

        question = st.chat_input(
            "Ask something about the video..."
        )

        # Invoke the RAG chain only after a valid question is submitted.
        if question and question.strip():

            question = question.strip()

            with st.chat_message("user"):
                st.markdown(question)

            try:
                with st.chat_message("assistant"):
                    with st.spinner("Searching the transcript..."):

                        response = st.session_state.rag_chain.invoke({
                            "question": question,
                            "chat_history": format_recent_chat_history(
                                st.session_state.chat_history
                            ),
                        })

                        answer = response_to_text(response)

                    if not answer:
                        answer = "I couldn't generate a response. Please try again."

                    st.markdown(answer)

                st.session_state.chat_history.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ])

            except Exception as e:
                import traceback

                st.error(f"Could not generate an answer: {e}")

                st.subheader("Complete Error Traceback")

                st.code(
                    traceback.format_exc(),
                    language="python"
                )

                print("FULL ERROR TRACEBACK:")
                traceback.print_exc()
    else:
        st.info(
            "Load the video first to start asking questions."
        )

    # --------------------------------------------------------
    # STUDY MATERIAL GENERATION
    # --------------------------------------------------------

    st.divider()

    st.subheader("Generate complete study material")

    st.write(
        "Create comprehensive study notes with video-based explanations, "
        "additional conceptual knowledge, fresh illustrative examples, "
        "worked calculations, highlighted formulas and topic-specific diagrams."
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
                f"{len(diagrams)} diagram(s) included."
            )

        except Exception as exc:
            import traceback

            st.error(
                f"Study material generation failed: {exc}"
            )

            st.code(
                traceback.format_exc(),
                language="text",
            )

    # --------------------------------------------------------
    # NOTES PREVIEW
    # --------------------------------------------------------

    if st.session_state.generated_notes:

        st.divider()

        st.subheader("Study notes preview")

        st.markdown(
            st.session_state.generated_notes
        )

        if st.session_state.generated_diagrams:

            st.subheader("Generated educational diagrams")

            for diagram in st.session_state.generated_diagrams:

                st.markdown(
                    f"**{diagram.get('title', 'Diagram')}**"
                )

                if diagram.get("description"):
                    st.caption(
                        diagram["description"]
                    )

                if diagram.get("image"):
                    st.image(
                        diagram["image"],
                        use_container_width=True,
                    )

        else:
            st.info(
                "No additional diagrams were needed "
                "for the generated notes."
            )

    # --------------------------------------------------------
    # PDF DOWNLOAD
    # --------------------------------------------------------

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
