
import re
import json

from app import (
    load_transcript,
    get_llm,
    extract_json_from_response,
    response_to_text,
    SUPPORTED_LANGUAGES,
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_QUESTIONS = 20
MAX_TRANSCRIPT_CHARS = 18000
MAX_NOTES_CHARS = 24000

QUESTION_TYPES = {
    "mcq",
    "multi_select",
    "true_false",
    "assertion_reason",
    "fill_blank",
    "scenario",
}

DIFFICULTIES = {
    "Easy",
    "Medium",
    "Hard",
    "Mixed",
}


# ============================================================
# CONTEXT BUILDERS
# ============================================================

def build_transcript_context(documents):
    """
    Sample transcript sections across the complete video.
    This prevents long videos from being represented only
    by their opening section.
    """

    full_text = "\n\n".join(
        doc.page_content.strip()
        for doc in documents
        if doc.page_content.strip()
    )

    if len(full_text) <= MAX_TRANSCRIPT_CHARS:
        return full_text

    window_size = MAX_TRANSCRIPT_CHARS // 4

    positions = [
        0,
        max(0, len(full_text) // 3 - window_size // 2),
        max(0, (2 * len(full_text)) // 3 - window_size // 2),
        max(0, len(full_text) - window_size),
    ]

    excerpts = []
    used_positions = set()

    for position in positions:
        if position in used_positions:
            continue

        used_positions.add(position)

        excerpt = full_text[
            position:position + window_size
        ].strip()

        if excerpt:
            excerpts.append(excerpt)

    return "\n\n[TRANSCRIPT SECTION BREAK]\n\n".join(excerpts)


def build_chat_context(chat_history, max_chars=6000):
    """
    Use chat history to understand learner interests.
    Conversation responses are not treated as verified facts.
    """

    if not isinstance(chat_history, list):
        return "No conversation history available."

    messages = []

    for message in chat_history[-20:]:

        if not isinstance(message, dict):
            continue

        role = str(message.get("role", ""))
        content = str(message.get("content", "")).strip()

        if role not in ("user", "assistant") or not content:
            continue

        messages.append(
            f"{role.upper()}: {content[:1200]}"
        )

    context = "\n\n".join(messages)

    return context[-max_chars:] if context else (
        "No conversation history available."
    )


# ============================================================
# LANGUAGE INSTRUCTIONS
# ============================================================

def get_language_instruction(language):

    instructions = {
        "English": """
Write every question, option, explanation, hint and report
insight in clear, simple English.
""",

        "Hindi": """
Write every question, option, explanation, hint and report
insight in natural Hindi using Devanagari script.
Standard technical terms may remain in English.
""",

        "Hinglish": """
Write every question, option, explanation, hint and report
insight in natural, readable Hinglish.
Keep standard technical terms in English.
""",
    }

    if language not in instructions:
        raise ValueError("Unsupported quiz language.")

    return instructions[language]


# ============================================================
# QUIZ GENERATION
# ============================================================

def generate_quiz(
    youtube_url,
    notes,
    chat_history,
    language="English",
    question_count=10,
    difficulty="Medium",
    question_types=None,
):
    """
    Generate a quiz using:
    1. YouTube transcript
    2. Generated study notes
    3. Learner's AI conversation

    Returns a validated quiz dictionary.
    """

    if not youtube_url or not youtube_url.strip():
        raise ValueError("A YouTube URL is required.")

    if not isinstance(notes, str) or not notes.strip():
        raise ValueError(
            "Generate study notes before creating a quiz."
        )

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError("Unsupported quiz language.")

    question_count = int(question_count)

    if not 5 <= question_count <= MAX_QUESTIONS:
        raise ValueError(
            f"Question count must be between 5 and {MAX_QUESTIONS}."
        )

    if difficulty not in DIFFICULTIES:
        raise ValueError("Invalid quiz difficulty.")

    if not question_types:
        question_types = ["mcq", "true_false"]

    question_types = list(dict.fromkeys(question_types))

    if (
        not question_types
        or not set(question_types).issubset(QUESTION_TYPES)
    ):
        raise ValueError("Invalid question types selected.")

    # --------------------------------------------------------
    # LOAD TRANSCRIPT
    # --------------------------------------------------------

    documents = load_transcript(youtube_url)

    transcript = build_transcript_context(documents)

    if not transcript:
        raise ValueError("The YouTube transcript is empty.")

    # --------------------------------------------------------
    # BUILD SOURCE CONTEXT
    # --------------------------------------------------------

    notes_context = notes[:MAX_NOTES_CHARS]

    chat_context = build_chat_context(chat_history)

    prompt = f"""
You are an expert educational assessment designer.

Create a high-quality quiz based on the supplied video
transcript, generated notes and learner conversation.

LANGUAGE:
{get_language_instruction(language)}

QUESTION COUNT:
Exactly {question_count}

DIFFICULTY:
{difficulty}

ALLOWED QUESTION TYPES:
{json.dumps(question_types)}

SOURCE PRIORITY:
1. YouTube transcript is the primary factual source.
2. Generated notes provide structured topic coverage.
3. Chat history helps identify learner interests.

SOURCE RULES:
- Questions must be supported by the transcript or notes.
- Do not invent video-specific facts.
- Do not treat previous assistant responses as verified facts.
- Use conversation history to prioritize topics, not invent facts.
- Do not introduce unrelated general-knowledge questions.
- Avoid duplicate questions.
- Cover different topics from the video.
- Preserve formulas, units and numerical values.

QUESTION QUALITY:
- Questions must be clear and unambiguous.
- Include recall, conceptual understanding and application.
- Match the requested difficulty.
- Use plausible distractors.
- Do not reveal the answer through question wording.

QUESTION TYPE RULES:

MCQ:
- Four options: A, B, C, D.
- Exactly one correct answer.

MULTIPLE SELECT:
- Four options.
- Two or three correct answers.
- Ask the learner to select all correct answers.

TRUE/FALSE:
- One factual statement.
- Options must have IDs "true" and "false".
- Exactly one correct answer.

ASSERTION/REASON:
- Include Assertion and Reason.
- Four options:
  A: Both true, and Reason correctly explains Assertion.
  B: Both true, but Reason does not explain Assertion.
  C: Assertion true, Reason false.
  D: Assertion false, Reason true.
- Exactly one correct answer.

FILL IN THE BLANK:
- One meaningful blank shown as ______.
- Expected answer must be a specific term, value or phrase.
- Include accepted equivalent answers.

SCENARIO:
- Give a realistic scenario based on the video.
- Test application of a concept.
- Four options and exactly one correct answer.

EXPLANATIONS:
- Explain why the correct answer is correct.
- Explain every incorrect option.
- Do not use explanations such as only "Correct" or "Incorrect".
- Include relevant reasoning or formulas where useful.

EVIDENCE:
- Every question must include a short evidence field.
- Evidence must be supported by the transcript or notes.
- Do not fabricate timestamps or quotations.
- evidence_source must be "transcript" or "notes".

RETURN VALID JSON ONLY:

{{
  "title": "Quiz title",
  "questions": [
    {{
      "id": "q1",
      "type": "mcq",
      "topic": "Concept name",
      "difficulty": "Medium",
      "question": "Question text",
      "options": [
        {{
          "id": "A",
          "text": "Option A",
          "explanation": "Option explanation"
        }},
        {{
          "id": "B",
          "text": "Option B",
          "explanation": "Option explanation"
        }},
        {{
          "id": "C",
          "text": "Option C",
          "explanation": "Option explanation"
        }},
        {{
          "id": "D",
          "text": "Option D",
          "explanation": "Option explanation"
        }}
      ],
      "correct_answer": ["B"],
      "accepted_answers": [],
      "explanation": "Complete answer explanation",
      "evidence": "Relevant source-supported fact",
      "evidence_source": "notes",
      "hint": "A useful clue"
    }}
  ]
}}

For true_false, use option IDs "true" and "false".

For fill_blank:
- options must be an empty list.
- correct_answer must contain the expected answer.
- accepted_answers may contain equivalent answers.

TRANSCRIPT:
{transcript}

GENERATED NOTES:
{notes_context}

LEARNER CONVERSATION:
{chat_context}

Generate exactly {question_count} questions.
Return JSON only, without Markdown fences.
"""

    response = get_llm().invoke(prompt)

    result = extract_json_from_response(
        response_to_text(response)
    )

    if not isinstance(result, dict):
        raise ValueError("The AI returned an invalid quiz format.")

    questions = result.get("questions", [])

    if not isinstance(questions, list):
        raise ValueError("Quiz questions are missing.")

    if len(questions) != question_count:
        raise ValueError(
            f"Expected {question_count} questions, "
            f"received {len(questions)}. Please regenerate."
        )

    # --------------------------------------------------------
    # VALIDATE QUESTIONS
    # --------------------------------------------------------

    validated = []
    seen_ids = set()

    for index, question in enumerate(questions, start=1):

        if not isinstance(question, dict):
            raise ValueError(f"Question {index} is invalid.")

        qid = str(question.get("id", f"q{index}")).strip()
        qtype = str(question.get("type", "")).strip()
        question_text = str(
            question.get("question", "")
        ).strip()

        if not qid or qid in seen_ids:
            raise ValueError("Question IDs must be unique.")

        seen_ids.add(qid)

        if qtype not in question_types:
            raise ValueError(
                f"Question {index} has an unrequested type."
            )

        if not question_text:
            raise ValueError(
                f"Question {index} has empty question text."
            )

        options = question.get("options", [])
        correct = question.get("correct_answer", [])
        accepted = question.get("accepted_answers", [])

        if not isinstance(options, list):
            raise ValueError("Question options must be a list.")

        if not isinstance(correct, list) or not correct:
            raise ValueError("A correct answer is missing.")

        if not isinstance(accepted, list):
            accepted = []

        cleaned_options = []
        option_ids = []

        for option in options:

            if not isinstance(option, dict):
                continue

            option_id = str(option.get("id", "")).strip()
            option_text = str(option.get("text", "")).strip()
            option_explanation = str(
                option.get("explanation", "")
            ).strip()

            if not option_id or not option_text:
                continue

            option_ids.append(option_id)

            cleaned_options.append({
                "id": option_id,
                "text": option_text,
                "explanation": option_explanation,
            })

        if len(set(option_ids)) != len(option_ids):
            raise ValueError("Duplicate option IDs found.")

        correct = [str(value) for value in correct]

        if qtype == "fill_blank":

            if cleaned_options:
                raise ValueError(
                    "Fill-in-the-blank must not contain options."
                )

        elif qtype == "true_false":

            if (
                set(option_ids) != {"true", "false"}
                or len(cleaned_options) != 2
                or len(correct) != 1
                or correct[0] not in {"true", "false"}
            ):
                raise ValueError(
                    f"True/False question {index} is malformed."
                )

        else:

            if len(cleaned_options) != 4:
                raise ValueError(
                    f"Question {index} must contain four options."
                )

            if not set(correct).issubset(set(option_ids)):
                raise ValueError(
                    f"Question {index} contains an invalid answer key."
                )

            if qtype in (
                "mcq",
                "scenario",
                "assertion_reason",
            ) and len(correct) != 1:

                raise ValueError(
                    f"Question {index} must have one correct answer."
                )

            if qtype == "multi_select" and not 2 <= len(correct) <= 3:

                raise ValueError(
                    f"Question {index} needs two or three correct answers."
                )

        explanation = str(
            question.get("explanation", "")
        ).strip()

        evidence = str(
            question.get("evidence", "")
        ).strip()

        if not explanation or not evidence:
            raise ValueError(
                f"Question {index} needs an explanation and evidence."
            )

        evidence_source = str(
            question.get("evidence_source", "notes")
        ).lower()

        if evidence_source not in {"transcript", "notes"}:
            evidence_source = "notes"

        validated.append({
            "id": qid,
            "type": qtype,
            "topic": str(
                question.get("topic", "General")
            ).strip() or "General",
            "difficulty": str(
                question.get("difficulty", difficulty)
            ),
            "question": question_text,
            "options": cleaned_options,
            "correct_answer": correct,
            "accepted_answers": [
                str(value) for value in accepted
            ],
            "explanation": explanation,
            "evidence": evidence,
            "evidence_source": evidence_source,
            "hint": str(
                question.get("hint", "")
            ).strip(),
        })

    return {
        "title": str(
            result.get("title", "Video Learning Quiz")
        ),
        "difficulty": difficulty,
        "language": language,
        "questions": validated,
    }


# ============================================================
# ANSWER EVALUATION
# ============================================================

def normalize_answer(value):

    return re.sub(
        r"\s+",
        " ",
        str(value or "").strip().casefold(),
    )


def evaluate_quiz_attempt(quiz, submitted_answers):
    """
    Evaluate answers without asking an LLM to calculate scores.
    """

    questions = quiz.get("questions", [])

    if not questions:
        raise ValueError("No quiz questions available.")

    results = []
    topic_stats = {}
    type_stats = {}

    correct_count = 0

    for question in questions:

        qid = question["id"]
        qtype = question["type"]

        submitted = submitted_answers.get(qid)

        if qtype == "fill_blank":

            submitted_text = normalize_answer(submitted)

            accepted = (
                question.get("accepted_answers", [])
                + question["correct_answer"]
            )

            accepted_set = {
                normalize_answer(value)
                for value in accepted
            }

            is_correct = (
                bool(submitted_text)
                and submitted_text in accepted_set
            )

        else:

            if not isinstance(submitted, list):
                submitted = (
                    [submitted] if submitted is not None else []
                )

            submitted_set = {
                str(value) for value in submitted
            }

            correct_set = {
                str(value)
                for value in question["correct_answer"]
            }

            is_correct = (
                bool(submitted_set)
                and submitted_set == correct_set
            )

        if is_correct:
            correct_count += 1

        topic = question.get("topic", "General")

        if topic not in topic_stats:
            topic_stats[topic] = {
                "total": 0,
                "correct": 0,
            }

        topic_stats[topic]["total"] += 1

        if is_correct:
            topic_stats[topic]["correct"] += 1

        if qtype not in type_stats:
            type_stats[qtype] = {
                "total": 0,
                "correct": 0,
            }

        type_stats[qtype]["total"] += 1

        if is_correct:
            type_stats[qtype]["correct"] += 1

        results.append({
            "question_id": qid,
            "question": question["question"],
            "type": qtype,
            "topic": topic,
            "submitted_answer": submitted,
            "correct_answer": question["correct_answer"],
            "accepted_answers": question["accepted_answers"],
            "is_correct": is_correct,
            "explanation": question["explanation"],
            "evidence": question["evidence"],
            "evidence_source": question["evidence_source"],
            "options": question["options"],
        })

    total = len(questions)
    incorrect_count = total - correct_count

    score = round(
        correct_count / total * 100,
        2,
    )

    if score >= 85:
        performance = "Excellent"
    elif score >= 70:
        performance = "Good"
    elif score >= 50:
        performance = "Developing"
    else:
        performance = "Needs Revision"

    topic_report = []

    for topic, stats in topic_stats.items():

        accuracy = round(
            stats["correct"] / stats["total"] * 100,
            2,
        )

        topic_report.append({
            "topic": topic,
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": accuracy,
        })

    topic_report.sort(
        key=lambda item: item["accuracy"],
        reverse=True,
    )

    type_report = []

    for qtype, stats in type_stats.items():

        accuracy = round(
            stats["correct"] / stats["total"] * 100,
            2,
        )

        type_report.append({
            "type": qtype,
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": accuracy,
        })

    return {
        "total_questions": total,
        "correct_answers": correct_count,
        "incorrect_answers": incorrect_count,
        "score": score,
        "performance": performance,
        "topic_report": topic_report,
        "type_report": type_report,
        "strong_topics": [
            item["topic"]
            for item in topic_report
            if item["accuracy"] >= 80
        ],
        "weak_topics": [
            item["topic"]
            for item in topic_report
            if item["accuracy"] < 60
        ],
        "results": results,
    }
