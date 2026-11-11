
"""
quiz_generation.py
------------------
Quiz generation, answer evaluation, and performance reporting
for the YT-RAG-Assist project.

Features:
- Six supported question types
- Explicit question-type assignment per position
- Small-batch generation
- Better JSON extraction and validation
- Retry prompts containing actual validation errors
- Exact and similarity-based duplicate question detection
- Previous-question exclusion support
- Representative transcript and notes selection
- Complete standalone questions
- Improved duplicate-option detection
- Improved calculation-question validation
- Answer evaluation
- Topic-wise performance reports
"""

import json
import re
import string
import unicodedata

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any


# ================================================================
# CONFIGURATION
# ================================================================

SUPPORTED_QUESTION_TYPES = {
    "mcq",
    "true_false",
    "assertion_reason",
    "multiple_select",
    "fill_blank",
    "scenario",
}

QUESTION_TYPE_LABELS = {
    "mcq": "Multiple Choice",
    "true_false": "True / False",
    "assertion_reason": "Assertion–Reason",
    "multiple_select": "Multiple Select",
    "fill_blank": "Fill in the Blank",
    "scenario": "Scenario Based",
}

ASSERTION_REASON_OPTIONS = [
    {
        "id": "A",
        "text": (
            "Both Assertion and Reason are true, "
            "and Reason correctly explains Assertion."
        ),
    },
    {
        "id": "B",
        "text": (
            "Both Assertion and Reason are true, "
            "but Reason does not correctly explain Assertion."
        ),
    },
    {
        "id": "C",
        "text": "Assertion is true, but Reason is false.",
    },
    {
        "id": "D",
        "text": "Assertion is false, but Reason is true.",
    },
]

TRUE_FALSE_OPTIONS = [
    {"id": "true", "text": "True"},
    {"id": "false", "text": "False"},
]

BATCH_SIZE = 3
MAX_BATCH_RETRIES = 3

MAX_TRANSCRIPT_CHARS = 14000
MAX_NOTES_CHARS = 10000
MAX_HISTORY_CHARS = 300

MIN_QUESTION_WORDS = 3

SEQUENCE_SIMILARITY_THRESHOLD = 0.88
TOKEN_SIMILARITY_THRESHOLD = 0.78


class QuizGenerationError(Exception):
    """Raised when quiz generation or validation fails."""


# ================================================================
# TEXT AND JSON HELPERS
# ================================================================

def _response_to_text(response: Any) -> str:
    """Convert different LLM response formats into plain text."""

    if isinstance(response, str):
        return response.strip()

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                if item.get("text"):
                    parts.append(str(item["text"]))

        return "\n".join(parts).strip()

    return str(content).strip() if content is not None else ""


def _extract_json_object(text: str) -> dict:
    """Extract one complete JSON object without guessing missing data."""

    if not text or not text.strip():
        raise QuizGenerationError(
            "The AI returned an empty response."
        )

    cleaned = text.strip().lstrip("\ufeff")

    cleaned = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```\s*$",
        "",
        cleaned,
    )

    try:
        parsed = json.loads(cleaned)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")

    if start < 0:
        raise QuizGenerationError(
            "The AI response did not contain a JSON object."
        )

    depth = 0
    in_string = False
    escaped = False
    end = None

    for index in range(start, len(cleaned)):
        char = cleaned[index]

        if in_string:

            if escaped:
                escaped = False

            elif char == "\\":
                escaped = True

            elif char == '"':
                in_string = False

            continue

        if char == '"':
            in_string = True

        elif char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth == 0:
                end = index + 1
                break

    if end is None:
        raise QuizGenerationError(
            "The AI response contains incomplete JSON."
        )

    try:
        parsed = json.loads(cleaned[start:end])

    except json.JSONDecodeError as exc:
        raise QuizGenerationError(
            f"Invalid JSON near line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(parsed, dict):
        raise QuizGenerationError(
            "Generated quiz JSON must be an object."
        )

    return parsed


def _clean_text(value: Any) -> str:

    if value is None:
        return ""

    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            ensure_ascii=False,
        )

    return str(value).strip()


def _normalize_answer(value: Any) -> str:

    text = _clean_text(value).lower()

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def _normalize_option_text(value: Any) -> str:
    """
    Normalize option text for duplicate detection.

    Preserves meaningful programming symbols such as:
    +, #, ., / and -.

    This is deliberately different from answer normalization.
    """

    text = unicodedata.normalize(
        "NFKC",
        _clean_text(value),
    ).casefold()

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    text = re.sub(
        r"[.!?,;:]+$",
        "",
        text,
    )

    text = re.sub(
        r"\s*/\s*",
        "/",
        text,
    )

    return text.strip()


def _prepare_chat_history(chat_history: Any) -> str:
    """
    Chat history identifies learner interests.
    It must never be treated as factual evidence.
    """

    if not chat_history:
        return "No previous chat history available."

    if isinstance(chat_history, str):
        return chat_history[-MAX_HISTORY_CHARS:]

    lines = []

    if isinstance(chat_history, list):

        for message in chat_history[-20:]:

            if isinstance(message, dict):

                role = message.get("role", "user")

                content = (
                    message.get("content")
                    or message.get("message")
                    or message.get("text")
                    or ""
                )

                if isinstance(content, list):
                    content = " ".join(
                        _clean_text(item)
                        for item in content
                    )

                if content:
                    lines.append(
                        f"{role}: {_clean_text(content)[:500]}"
                    )

            else:
                lines.append(
                    _clean_text(message)[:500]
                )

    return (
        "\n".join(lines)[-MAX_HISTORY_CHARS:]
        or "No previous chat history available."
    )


# ================================================================
# REPRESENTATIVE SOURCE SELECTION
# ================================================================

def _select_representative_text(
    text: str,
    max_chars: int,
) -> str:
    """
    Select text from across the complete source.
    Avoids taking only the beginning of long transcripts.
    """

    text = _clean_text(text)

    if not text:
        return ""

    if len(text) <= max_chars:
        return text

    section_count = 7

    section_size = max(
        1,
        max_chars // section_count,
    )

    selected_parts = []

    for index in range(section_count):

        center = (
            (index + 0.5)
            * len(text)
            / section_count
        )

        start = max(
            0,
            int(center - section_size / 2),
        )

        end = min(
            len(text),
            start + section_size,
        )

        if start > 0:
            next_space = text.find(" ", start)

            if next_space != -1 and next_space < start + 150:
                start = next_space + 1

        if end < len(text):
            previous_space = text.rfind(" ", start, end)

            if previous_space > start:
                end = previous_space

        part = text[start:end].strip()

        if part:
            selected_parts.append(part)

    unique_parts = []
    seen_parts = set()

    for part in selected_parts:

        normalized = _normalize_answer(part)

        if normalized and normalized not in seen_parts:
            seen_parts.add(normalized)
            unique_parts.append(part)

    return "\n\n[...source section omitted...]\n\n".join(
        unique_parts
    )[:max_chars]


# ================================================================
# QUESTION NORMALIZATION
# ================================================================

def _normalize_question_type(question_type: Any) -> str:

    value = (
        _clean_text(question_type)
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    aliases = {
        "multiple_choice": "mcq",
        "multiple_choice_question": "mcq",
        "mcqs": "mcq",
        "truefalse": "true_false",
        "tf": "true_false",
        "assertion_reasoning": "assertion_reason",
        "assertion_and_reason": "assertion_reason",
        "multi_select": "multiple_select",
        "select_all": "multiple_select",
        "fill_in_the_blank": "fill_blank",
        "fill_in_blank": "fill_blank",
        "scenario_based": "scenario",
        "case_based": "scenario",
    }

    return aliases.get(value, value)


def _normalize_options(options: Any) -> list:

    if not isinstance(options, list):
        return []

    normalized = []

    for index, option in enumerate(options):

        if isinstance(option, dict):

            option_id = _clean_text(
                option.get("id")
                or option.get("key")
                or option.get("label")
                or chr(65 + index)
            )

            option_text = _clean_text(
                option.get("text")
                or option.get("value")
                or option.get("label")
            )

        else:

            option_id = chr(65 + index)
            option_text = _clean_text(option)

        if option_id and option_text:

            normalized.append({
                "id": option_id,
                "text": option_text,
            })

    return normalized


def _normalize_correct_answer(
    value: Any,
    question_type: str,
) -> Any:

    if question_type == "multiple_select":

        if not isinstance(value, list):
            value = [value]

        return sorted({
            _clean_text(item).upper()
            for item in value
            if _clean_text(item)
        })

    if question_type == "fill_blank":

        values = (
            value
            if isinstance(value, list)
            else [value]
        )

        return [
            _clean_text(item)
            for item in values
            if _clean_text(item)
        ]

    return _clean_text(value).lower()


# ================================================================
# QUESTION DUPLICATE DETECTION
# ================================================================

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "of", "to", "in", "on", "at", "for", "from",
    "and", "or", "but", "by", "with", "as", "what",
    "which", "who", "when", "where", "why", "how",
    "does", "do", "did", "can", "could", "would",
    "should", "will", "has", "have", "had",
    "explain", "describe", "identify", "determine",
    "following", "given", "correct", "main",
}


def _question_signature(question: dict) -> str:

    return _normalize_answer(
        question.get("question", "")
    )


def _question_tokens(question_text: str) -> set:

    normalized = _normalize_answer(question_text)

    return {
        token
        for token in normalized.split()
        if len(token) > 2 and token not in STOP_WORDS
    }


def _question_similarity(
    first: str,
    second: str,
) -> float:

    first_normalized = _normalize_answer(first)
    second_normalized = _normalize_answer(second)

    if not first_normalized or not second_normalized:
        return 0.0

    if first_normalized == second_normalized:
        return 1.0

    sequence_score = SequenceMatcher(
        None,
        first_normalized,
        second_normalized,
    ).ratio()

    first_tokens = _question_tokens(first)
    second_tokens = _question_tokens(second)

    if not first_tokens or not second_tokens:
        return sequence_score

    intersection = len(
        first_tokens.intersection(second_tokens)
    )

    union = len(
        first_tokens.union(second_tokens)
    )

    jaccard_score = (
        intersection / union
        if union
        else 0.0
    )

    smaller_size = min(
        len(first_tokens),
        len(second_tokens),
    )

    containment_score = (
        intersection / smaller_size
        if smaller_size
        else 0.0
    )

    combined_score = max(
        sequence_score,
        (
            0.65 * jaccard_score
            + 0.35 * containment_score
        ),
    )

    return combined_score


def _find_duplicate_question(
    question_text: str,
    existing_questions: list,
) -> str | None:

    normalized = _normalize_answer(question_text)

    for existing in existing_questions:

        existing_text = (
            existing
            if isinstance(existing, str)
            else _clean_text(
                existing.get("question", "")
            )
            if isinstance(existing, dict)
            else ""
        )

        if not existing_text:
            continue

        existing_normalized = _normalize_answer(
            existing_text
        )

        if normalized == existing_normalized:
            return existing_text

        similarity = _question_similarity(
            question_text,
            existing_text,
        )

        if similarity >= SEQUENCE_SIMILARITY_THRESHOLD:
            return existing_text

        first_tokens = _question_tokens(question_text)
        second_tokens = _question_tokens(existing_text)

        if first_tokens and second_tokens:

            overlap = (
                len(first_tokens.intersection(second_tokens))
                / len(first_tokens.union(second_tokens))
            )

            if (
                overlap >= TOKEN_SIMILARITY_THRESHOLD
                and similarity >= 0.68
            ):
                return existing_text

    return None


# ================================================================
# QUESTION VALIDATION
# ================================================================

def _validate_question_wording(
    question_text: str,
    question_type: str,
    index: int,
) -> None:
    """Validate completeness and clarity of generated question text."""

    text = re.sub(
        r"\s+",
        " ",
        question_text,
    ).strip()

    lowered = text.lower()
    words = text.split()

    minimum_words = {
        "mcq": 3,
        "true_false": 3,
        "assertion_reason": 6,
        "multiple_select": 3,
        "fill_blank": 3,
        "scenario": 5,
    }

    required_words = minimum_words.get(
        question_type,
        MIN_QUESTION_WORDS,
    )

    if len(words) < required_words:
        raise QuizGenerationError(
            f"Question {index + 1} is incomplete. "
            f"It needs at least {required_words} words "
            f"for the {question_type} question type."
        )

    vague_patterns = [
        r"what was this example about\??",
        r"what is discussed here\??",
        r"what was explained\??",
        r"which of the following is correct\??",
        r"what is the answer\??",
        r"what happened in the video\??",
        r"what did we learn\??",
        r"which statement is correct\??",
        r"identify the correct option\??",
        r"what is this\??",
    ]

    if any(
        re.fullmatch(pattern, lowered)
        for pattern in vague_patterns
    ):
        raise QuizGenerationError(
            f"Question {index + 1} is too vague."
        )

    unclear_references = (
        "this example",
        "the above example",
        "the given example",
        "this concept",
        "the above concept",
        "the following process",
        "as discussed earlier",
        "as mentioned above",
        "this process",
        "that process",
        "this method",
        "that method",
        "the previous example",
    )

    if any(
        reference in lowered
        for reference in unclear_references
    ):
        raise QuizGenerationError(
            f"Question {index + 1} contains an unclear reference."
        )

    if question_type == "assertion_reason":

        if (
            "assertion" not in lowered
            or "reason" not in lowered
        ):
            raise QuizGenerationError(
                f"Question {index + 1} must contain "
                "both Assertion and Reason."
            )

    if question_type == "fill_blank":

        blank_patterns = [
            r"_{2,}",
            r"\[\s*blank\s*\]",
            r"<blank>",
            r"□{1,8}",
        ]

        has_blank = any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in blank_patterns
        )

        if not has_blank:
            raise QuizGenerationError(
                f"Question {index + 1} must contain "
                "a clearly marked blank."
            )

    # ------------------------------------------------------------
    # IMPROVED CALCULATION VALIDATION
    # ------------------------------------------------------------
    #
    # Previous logic:
    # It required at least two numbers whenever a question
    # contained words such as "calculate" or "weighted sum".
    #
    # This incorrectly rejected:
    # - Questions requiring only one numerical input.
    # - Conceptual questions mentioning calculation terminology.
    # - Questions that ask for a result without using the old
    #   exact calculation phrases.
    #
    # New logic:
    # 1. Detect an explicit calculation action.
    # 2. Detect whether the question asks for a numerical target.
    # 3. Only flag it when no numerical input is present.
    # 4. Do not impose an arbitrary minimum of two numbers.
    #
    # This is a basic completeness check, not a mathematical
    # proof that every required formula input is present.

    calculation_actions = (
        "calculate",
        "compute",
        "determine",
        "evaluate",
        "solve",
        "find",
    )

    calculation_targets = (
        "value",
        "numerical value",
        "output",
        "result",
        "weighted sum",
        "net input",
        "pre-activation",
        "sum",
        "average",
        "mean",
        "percentage",
        "ratio",
        "distance",
        "speed",
        "probability",
        "total",
    )

    has_calculation_action = any(
        re.search(
            rf"\b{re.escape(action)}\b",
            lowered,
        )
        for action in calculation_actions
    )

    has_calculation_target = any(
        target in lowered
        for target in calculation_targets
    )

    # Supports:
    # 5
    # -2
    # 3.14
    # .75
    # 2e-3
    # -1.5E+2
    numeric_values = re.findall(
        r"(?<![A-Za-z_])"
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][+-]?\d+)?"
        r"(?![A-Za-z_])",
        text,
    )

    if (
        has_calculation_action
        and has_calculation_target
        and len(numeric_values) == 0
    ):
        raise QuizGenerationError(
            f"Question {index + 1} may be missing numerical inputs."
        )


def _validate_question(
    question: dict,
    index: int,
) -> dict:

    if not isinstance(question, dict):
        raise QuizGenerationError(
            f"Question {index + 1} has an invalid format."
        )

    question_type = _normalize_question_type(
        question.get("type")
        or question.get("question_type")
        or "mcq"
    )

    if question_type not in SUPPORTED_QUESTION_TYPES:
        raise QuizGenerationError(
            f"Question {index + 1} has unsupported type: "
            f"{question_type}."
        )

    question_text = ""

    for field in (
        "question",
        "text",
        "question_text",
        "question_statement",
        "stem",
    ):
        candidate = _clean_text(question.get(field))

        if candidate:
            question_text = candidate
            break

    if not question_text:
        available_fields = ", ".join(
            str(key) for key in question.keys()
        )

        raise QuizGenerationError(
            f"Question {index + 1} has no question text. "
            f"Available fields: {available_fields or 'none'}."
        )

    _validate_question_wording(
        question_text,
        question_type,
        index,
    )

    options = _normalize_options(
        question.get("options", [])
    )

    raw_answer = question.get(
        "correct_answer",
        question.get("answer"),
    )

    correct_answer = _normalize_correct_answer(
        raw_answer,
        question_type,
    )

    if question_type in {
        "mcq",
        "scenario",
        "multiple_select",
    }:

        if len(options) != 4:
            raise QuizGenerationError(
                f"Question {index + 1} must have exactly four options."
            )

        option_ids = [
            item["id"].casefold()
            for item in options
        ]

        if len(set(option_ids)) != 4:
            raise QuizGenerationError(
                f"Question {index + 1} has duplicate option IDs."
            )

        option_texts = [
            _normalize_option_text(item["text"])
            for item in options
        ]

        if any(not text for text in option_texts):
            raise QuizGenerationError(
                f"Question {index + 1} contains an empty option."
            )

        if len(set(option_texts)) != 4:

            duplicates = sorted({
                text
                for text in option_texts
                if option_texts.count(text) > 1
            })

            raise QuizGenerationError(
                f"Question {index + 1} has duplicate options: "
                f"{', '.join(duplicates)}. "
                "All four options must be distinct."
            )

    if question_type in {"mcq", "scenario"}:

        valid_ids = {
            item["id"].casefold()
            for item in options
        }

        if correct_answer not in valid_ids:
            raise QuizGenerationError(
                f"Question {index + 1} has an invalid correct option."
            )

    elif question_type == "true_false":

        options = TRUE_FALSE_OPTIONS

        if correct_answer not in {"true", "false"}:
            raise QuizGenerationError(
                f"Question {index + 1} must have "
                "True or False as its answer."
            )

    elif question_type == "assertion_reason":

        options = ASSERTION_REASON_OPTIONS

        if correct_answer not in {"a", "b", "c", "d"}:
            raise QuizGenerationError(
                f"Question {index + 1} must have "
                "A, B, C or D as its answer."
            )

    elif question_type == "multiple_select":

        valid_ids = {
            item["id"].upper()
            for item in options
        }

        if not isinstance(correct_answer, list):
            raise QuizGenerationError(
                f"Question {index + 1} needs multiple correct answers."
            )

        if not 2 <= len(correct_answer) <= 3:
            raise QuizGenerationError(
                f"Question {index + 1} must have "
                "two or three correct answers."
            )

        if len(correct_answer) != len(set(correct_answer)):
            raise QuizGenerationError(
                f"Question {index + 1} has duplicate correct answers."
            )

        if not set(correct_answer).issubset(valid_ids):
            raise QuizGenerationError(
                f"Question {index + 1} has invalid correct options."
            )

    elif question_type == "fill_blank":

        if not correct_answer:
            raise QuizGenerationError(
                f"Question {index + 1} has no accepted answer."
            )

        options = []

    explanation = _clean_text(
        question.get("explanation")
    )

    if len(explanation.split()) < 5:
        raise QuizGenerationError(
            f"Question {index + 1} needs a meaningful explanation."
        )

    return {
        "id": (
            _clean_text(question.get("id"))
            or f"Q{index + 1}"
        ),
        "type": question_type,
        "type_label": QUESTION_TYPE_LABELS[question_type],
        "question": question_text,
        "options": options,
        "correct_answer": correct_answer,
        "accepted_answers": (
            correct_answer
            if question_type == "fill_blank"
            else []
        ),
        "explanation": explanation,
        "topic": (
            _clean_text(question.get("topic"))
            or "General"
        ),
        "difficulty": (
            _clean_text(question.get("difficulty"))
            or "Medium"
        ),
        "evidence": _clean_text(
            question.get("evidence")
        ),
        "hint": _clean_text(
            question.get("hint")
        ),
    }


# ================================================================
# PROMPT GENERATION
# ================================================================

def _build_prompt(
    transcript_context: str,
    notes_context: str,
    history_context: str,
    language: str,
    question_count: int,
    difficulty: str,
    normalized_types: list,
    previous_questions: list | None = None,
) -> str:

    schema = {
        "title": "Topic-based quiz title",
        "instructions": "Short learner instructions",
        "questions": [
            {
                "id": "Q1",
                "type": "mcq",
                "question": "Complete standalone question",
                "options": [
                    {"id": "A", "text": "Option A"},
                    {"id": "B", "text": "Option B"},
                    {"id": "C", "text": "Option C"},
                    {"id": "D", "text": "Option D"},
                ],
                "correct_answer": "B",
                "explanation": (
                    "Explain why the correct answer is correct."
                ),
                "topic": "Topic name",
                "difficulty": "Medium",
                "evidence": "Supporting concept from source",
                "hint": "Short hint without revealing the answer",
            }
        ],
    }

    required_types = "\n".join(
        f"Question {index + 1}: {question_type}"
        for index, question_type in enumerate(normalized_types)
    )

    previous_questions = previous_questions or []

    previous_text = "\n".join(
        f"- {str(question)[:300]}"
        for question in previous_questions[-40:]
    )

    if not previous_text:
        previous_text = "No previous questions supplied."

    return f"""
You are an expert educational assessment designer.

Generate exactly {question_count} questions.

LEARNER SETTINGS

Language: {language}
Difficulty: {difficulty}

EXACT QUESTION TYPE ASSIGNMENT

Follow this assignment in the exact order.
Do not change, replace or rearrange question types.

{required_types}

SOURCE MATERIAL

YOUTUBE TRANSCRIPT:
{transcript_context}

GENERATED STUDY NOTES:
{notes_context}

LEARNER CHAT HISTORY:
{history_context}

PREVIOUSLY GENERATED QUESTIONS — DO NOT REPEAT THESE:

{previous_text}

FACTUAL RULES

1. Use transcript and notes as primary factual sources.
2. Chat history identifies learner interests only.
3. Never use chat history as factual evidence.
4. Never invent facts, formulas, values, weights, biases or events.
5. Questions must be relevant to the actual video.
6. Do not repeat questions using different wording.
7. Do not test the same specific fact or calculation twice.
8. Cover different concepts and sections of the supplied material.
9. Use the requested language consistently.
10. Provide meaningful explanations.
11. Give supporting evidence for each question.

QUESTION COMPLETENESS

1. Every question must be independently understandable.
2. Name the exact concept, component or principle being tested.
3. Never use vague references such as "this example" or "above".
4. Calculation questions must include all necessary input values.
5. Include all relevant weights, biases, equations and conditions.
6. For weighted-sum questions, include inputs, corresponding weights
   and bias whenever they are required.
7. A calculation may require one numerical input or several inputs.
   Include exactly the information required by the formula.
8. Do not assume every calculation requires two or more numbers.
9. If the source does not provide sufficient calculation data,
   ask a conceptual question instead.
10. Do not invent numerical values to complete a calculation.
11. Do not turn a conceptual question into a calculation unnecessarily.
12. Do not reveal the answer in the question.
13. True/False questions must be complete factual statements.
14. Assertion–Reason questions must explicitly state both statements.
15. Fill-in-the-blank questions must have one clearly defined blank.
16. Scenario questions must contain enough information to solve them.
17. Do not create incomplete questions simply to meet the count.

NUMERICAL QUESTION RULES

1. First decide whether the question genuinely requires a numerical result.
2. If it requires calculation, include every input needed to solve it.
3. Include formula-specific values, units, weights, biases and conditions.
4. Do not assume a calculation needs two or more numerical values.
5. If one input is sufficient, one input is acceptable.
6. If the source lacks the required numerical data, use a conceptual question.
7. Do not ask learners to calculate a value that cannot be determined
   from the question.
8. Verify that the question is independently solvable before returning it.

QUESTION TYPE RULES

MCQ:
- Exactly four options.
- Exactly one correct option ID.
- All four options must be distinct.

TRUE/FALSE:
- Complete factual statement.
- Answer must be "true" or "false".

ASSERTION–REASON:
- Explicit Assertion and Reason.
- A = Both true and Reason explains Assertion.
- B = Both true but Reason does not explain Assertion.
- C = Assertion true and Reason false.
- D = Assertion false and Reason true.

MULTIPLE SELECT:
- Exactly four options.
- Two or three correct option IDs.
- All four options must be distinct.

FILL IN THE BLANK:
- No options.
- Provide accepted answers as a list.
- Include exactly one clearly marked blank.

SCENARIO:
- A complete practical situation.
- Exactly four options.
- Exactly one correct option.
- All four options must be distinct.

OPTION QUALITY AND DUPLICATE PREVENTION

1. Every option must be non-empty.
2. Never repeat an option.
3. Do not create duplicates by changing capitalization,
   spacing or insignificant punctuation.
4. Do not use synonyms or equivalent answers as separate
   choices in single-answer questions.
5. Preserve meaningful technical distinctions such as C, C++ and C#.
6. Incorrect choices must be plausible and related to the subject.
7. Keep options similar in style and level of detail.
8. Do not make an option correct merely because it is more detailed.
9. Do not use multiple versions of the same answer.
10. Independently compare all four final options before returning JSON.

OUTPUT RULES

Return ONLY one valid JSON object.
Use double quotes for all JSON keys and strings.
No Markdown fences.
No commentary.
No trailing commas.
Generate exactly {question_count} questions.
Use only the assigned question types and exact order.

Before returning, verify that every question is complete,
all options are distinct, every answer is valid, and no question
repeats a previous question or another question in this batch.

Required JSON structure:

{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


# ================================================================
# BATCH GENERATION
# ================================================================

def _make_batches(
    items: list,
    batch_size: int,
) -> list:

    return [
        items[index:index + batch_size]
        for index in range(0, len(items), batch_size)
    ]


def _invoke_batch(
    llm,
    prompt: str,
    batch_number: int,
    batch_size: int,
) -> dict:

    try:

        response = llm.invoke(prompt)

        response_text = _response_to_text(response)

        parsed = _extract_json_object(response_text)

        questions = parsed.get("questions")

        if not isinstance(questions, list):
            raise QuizGenerationError(
                f"Batch {batch_number}: missing questions list."
            )

        if len(questions) != batch_size:
            raise QuizGenerationError(
                f"Batch {batch_number}: expected {batch_size} "
                f"questions but received {len(questions)}."
            )

        return parsed

    except QuizGenerationError:
        raise

    except Exception as exc:
        raise QuizGenerationError(
            f"LLM request failed for batch {batch_number}: {exc}"
        ) from exc


def _build_retry_instructions(last_error: str) -> str:
    """Build targeted correction instructions for a failed batch."""

    general_instructions = """
PREVIOUS VALIDATION FAILURE

{error}

IMPORTANT CORRECTIONS

1. Generate a completely new version of the invalid question.
2. Do not unnecessarily increase question length.
3. A short question is acceptable if complete and specific.
4. Identify the exact concept being tested.
5. Avoid incomplete sentences and vague references.
6. Include all information needed to answer the question.
7. Preserve the assigned question type and its position.
8. Verify the correct answer and explanation.
9. Do not remove required fields.
10. Return valid JSON only.

QUESTION FIELD REQUIREMENT

Every question object MUST contain a non-empty "question" field.
Do not omit the field or return null or an empty string.

OPTION REQUIREMENTS

Every option must be non-empty and meaningfully different.

- Do not repeat options.
- Do not use capitalization or spacing changes to create
  fake alternatives.
- Do not use synonyms or equivalent answers as separate
  choices in single-answer questions.
- Keep all options relevant to the same subject.
- Incorrect choices must be plausible but factually incorrect.
- Preserve meaningful technical distinctions such as C, C++
  and C#.
- Do not change the correct answer merely to avoid a
  duplicate-option validation error.

If the validation error identifies duplicate options,
replace the duplicated options with genuinely different,
factually valid distractors.

Return exactly the required number of questions.
"""

    numerical_error_instructions = ""

    error_lower = last_error.lower()

    if (
        "numerical input" in error_lower
        or "numerical result" in error_lower
        or "calculation" in error_lower
    ):
        numerical_error_instructions = """

NUMERICAL QUESTION CORRECTION

The previous question failed numerical completeness validation.

1. First decide whether the question genuinely requires a numerical result.
2. If it is a numerical problem, include every required input value.
3. Include all relevant formula values, weights, biases, units
   and conditions.
4. Do not invent values that are absent from the transcript or notes.
5. If sufficient numerical information is unavailable, replace
   the question with a conceptual question about the same topic.
6. Do not turn a conceptual question into a calculation unnecessarily.
7. Do not assume that every calculation requires two or more numbers.
8. One numerical input is acceptable when the formula requires only one.
9. Ensure the final question is independently solvable.
"""

    general_instructions += numerical_error_instructions

    return general_instructions.format(
        error=last_error
    )


def _generate_validated_batch(
    llm,
    transcript_context: str,
    notes_context: str,
    history_context: str,
    language: str,
    difficulty: str,
    batch_types: list,
    batch_number: int,
    existing_questions: list | None = None,
    previous_questions: list | None = None,
) -> tuple:

    batch_size = len(batch_types)

    existing_questions = existing_questions or []
    previous_questions = previous_questions or []

    all_excluded_questions = (
        list(existing_questions)
        + list(previous_questions)
    )

    last_error = None

    for attempt in range(MAX_BATCH_RETRIES):

        current_transcript = (
            transcript_context
            if attempt == 0
            else transcript_context[:9000]
        )

        current_notes = (
            notes_context
            if attempt == 0
            else notes_context[:6500]
        )

        current_history = (
            history_context
            if attempt == 0
            else "No additional chat context."
        )

        prompt = _build_prompt(
            transcript_context=current_transcript,
            notes_context=current_notes,
            history_context=current_history,
            language=language,
            question_count=batch_size,
            difficulty=difficulty,
            normalized_types=batch_types,
            previous_questions=all_excluded_questions,
        )

        if attempt > 0:
            prompt += _build_retry_instructions(last_error)

        try:

            parsed = _invoke_batch(
                llm=llm,
                prompt=prompt,
                batch_number=batch_number,
                batch_size=batch_size,
            )

            validated = []

            for index, raw_question in enumerate(
                parsed["questions"]
            ):

                raw_options = (
                    raw_question.get("options", [])
                    if isinstance(raw_question, dict)
                    else []
                )

                print(
                    f"[QUIZ DEBUG] Batch {batch_number}, "
                    f"Question {index + 1} raw options:",
                    [
                        (
                            option.get(
                                "text",
                                option.get(
                                    "value",
                                    option.get("label", ""),
                                ),
                            )
                            if isinstance(option, dict)
                            else option
                        )
                        for option in raw_options
                    ],
                )

                question = _validate_question(
                    raw_question,
                    index,
                )

                expected_type = batch_types[index]

                if question["type"] != expected_type:
                    raise QuizGenerationError(
                        f"Question {index + 1} in batch "
                        f"{batch_number} must be '{expected_type}', "
                        f"but the model generated "
                        f"'{question['type']}'."
                    )

                duplicate = _find_duplicate_question(
                    question["question"],
                    all_excluded_questions + validated,
                )

                if duplicate:
                    raise QuizGenerationError(
                        f"Question {index + 1} in batch "
                        f"{batch_number} is too similar to an "
                        f"existing question: {duplicate[:180]}"
                    )

                validated.append(question)

            title = _clean_text(
                parsed.get("title")
            )

            return validated, title

        except QuizGenerationError as exc:

            last_error = str(exc)

            print(
                f"[QUIZ DEBUG] Batch {batch_number}, "
                f"attempt {attempt + 1} failed: {last_error}"
            )

    raise QuizGenerationError(
        f"Batch {batch_number} failed after "
        f"{MAX_BATCH_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


# ================================================================
# MAIN QUIZ GENERATION
# ================================================================

def generate_quiz(
    llm,
    transcript: str,
    notes: str = "",
    chat_history: Any = None,
    language: str = "English",
    question_count: int = 10,
    difficulty: str = "Mixed",
    question_types: list | None = None,
    previous_questions: list | None = None,
) -> dict:

    """
    Generate a complete quiz using small validated batches.

    previous_questions:
        Previously generated question texts for this video.
        These are excluded from the new quiz.
    """

    transcript = _clean_text(transcript)
    notes = _clean_text(notes)

    if not transcript and not notes:
        raise QuizGenerationError(
            "No video transcript or study notes are available. "
            "Process a YouTube video first."
        )

    try:
        question_count = int(question_count)

    except (TypeError, ValueError):
        question_count = 10

    question_count = max(
        1,
        min(question_count, 30),
    )

    if not question_types:
        question_types = [
            "mcq",
            "true_false",
            "assertion_reason",
            "multiple_select",
            "fill_blank",
            "scenario",
        ]

    normalized_types = []

    for item in question_types:

        normalized = _normalize_question_type(item)

        if normalized in SUPPORTED_QUESTION_TYPES:
            normalized_types.append(normalized)

    normalized_types = list(
        dict.fromkeys(normalized_types)
    )

    if not normalized_types:
        normalized_types = ["mcq"]

    expected_types = [
        normalized_types[index % len(normalized_types)]
        for index in range(question_count)
    ]

    batches = _make_batches(
        expected_types,
        BATCH_SIZE,
    )

    transcript_context = _select_representative_text(
        transcript,
        MAX_TRANSCRIPT_CHARS,
    )

    notes_context = _select_representative_text(
        notes,
        MAX_NOTES_CHARS,
    )

    history_context = _prepare_chat_history(
        chat_history
    )[:MAX_HISTORY_CHARS]

    if isinstance(previous_questions, list):
        previous_questions = [
            _clean_text(item.get("question", ""))
            if isinstance(item, dict)
            else _clean_text(item)
            for item in previous_questions
        ]
    else:
        previous_questions = []

    previous_questions = [
        item
        for item in previous_questions
        if item
    ]

    all_questions = []
    seen_ids = set()

    quiz_title = ""

    for batch_index, batch_types in enumerate(
        batches,
        start=1,
    ):

        validated_batch, batch_title = (
            _generate_validated_batch(
                llm=llm,
                transcript_context=transcript_context,
                notes_context=notes_context,
                history_context=history_context,
                language=language,
                difficulty=difficulty,
                batch_types=batch_types,
                batch_number=batch_index,
                existing_questions=all_questions,
                previous_questions=previous_questions,
            )
        )

        if batch_title and not quiz_title:
            quiz_title = batch_title

        for question in validated_batch:

            question_id = f"Q{len(all_questions) + 1}"

            while question_id in seen_ids:
                question_id = f"Q{len(all_questions) + 1}"

            question["id"] = question_id

            seen_ids.add(question_id)

            all_questions.append(question)

    if len(all_questions) != question_count:
        raise QuizGenerationError(
            f"Quiz generation incomplete. Requested "
            f"{question_count}, generated {len(all_questions)}."
        )

    return {
        "title": quiz_title or "Video Learning Quiz",
        "instructions": (
            "Answer all questions and submit your quiz "
            "to see your results."
        ),
        "questions": all_questions,
        "total_questions": len(all_questions),
        "language": language,
        "difficulty": difficulty,
    }


# ================================================================
# ANSWER CHECKING
# ================================================================

def _check_answer(
    question: dict,
    submitted_answer: Any,
) -> bool:

    question_type = question["type"]

    correct_answer = question["correct_answer"]

    if submitted_answer is None:
        return False

    if question_type == "multiple_select":

        if not isinstance(submitted_answer, list):
            submitted_answer = [submitted_answer]

        submitted = sorted({
            _clean_text(item).upper()
            for item in submitted_answer
        })

        return submitted == correct_answer

    if question_type == "fill_blank":

        submitted = _normalize_answer(
            submitted_answer
        )

        accepted = [
            _normalize_answer(answer)
            for answer in question.get(
                "accepted_answers",
                [],
            )
        ]

        return submitted in accepted

    return (
        _clean_text(submitted_answer).lower()
        == correct_answer
    )


def _format_correct_answer(
    question: dict,
) -> Any:

    answer = question["correct_answer"]

    if question["type"] in {
        "fill_blank",
        "multiple_select",
    }:
        return answer

    return answer.upper()


# ================================================================
# QUIZ EVALUATION AND REPORT CARD
# ================================================================

def evaluate_quiz_attempt(
    quiz: dict,
    submitted_answers: dict,
    time_taken_seconds: int | float = 0,
) -> dict:

    """
    Evaluate submitted answers.

    Returns:
    - Score and percentage
    - Correct, incorrect and unanswered counts
    - Topic-wise performance
    - Strengths and weak topics
    - Explanations
    - Revision suggestions
    - Time taken
    """

    if not isinstance(quiz, dict):
        raise QuizGenerationError(
            "Invalid quiz data."
        )

    questions = quiz.get("questions", [])

    if not questions:
        raise QuizGenerationError(
            "There are no questions to evaluate."
        )

    if not isinstance(submitted_answers, dict):
        submitted_answers = {}

    results = []

    correct_count = 0
    incorrect_count = 0
    unanswered_count = 0

    topic_stats = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "incorrect": 0,
            "unanswered": 0,
        }
    )

    for question in questions:

        question_id = question["id"]

        question_type = question["type"]

        submitted = submitted_answers.get(
            question_id
        )

        is_unanswered = (
            submitted is None
            or submitted == ""
            or submitted == []
        )

        is_correct = (
            False
            if is_unanswered
            else _check_answer(
                question,
                submitted,
            )
        )

        if is_unanswered:

            status = "unanswered"
            unanswered_count += 1

        elif is_correct:

            status = "correct"
            correct_count += 1

        else:

            status = "incorrect"
            incorrect_count += 1

        topic = (
            question.get("topic")
            or "General"
        )

        topic_stats[topic]["total"] += 1
        topic_stats[topic][status] += 1

        results.append({
            "question_id": question_id,
            "question": question["question"],
            "type": question_type,
            "type_label": question.get(
                "type_label",
                question_type,
            ),
            "topic": topic,
            "submitted_answer": submitted,
            "correct_answer": _format_correct_answer(
                question
            ),
            "is_correct": is_correct,
            "status": status,
            "explanation": question.get(
                "explanation",
                "",
            ),
            "evidence": question.get(
                "evidence",
                "",
            ),
            "options": question.get(
                "options",
                [],
            ),
        })

    total_questions = len(questions)

    percentage = round(
        (correct_count / total_questions) * 100,
        2,
    )

    topic_report = []

    for topic, stats in topic_stats.items():

        topic_report.append({
            "topic": topic,
            "total": stats["total"],
            "correct": stats["correct"],
            "incorrect": stats["incorrect"],
            "unanswered": stats["unanswered"],
            "percentage": round(
                (
                    stats["correct"]
                    / stats["total"]
                ) * 100,
                2,
            ),
        })

    topic_report.sort(
        key=lambda item: (
            item["percentage"],
            item["total"],
        ),
        reverse=True,
    )

    strengths = [
        item["topic"]
        for item in topic_report
        if (
            item["percentage"] >= 75
            and item["total"] >= 2
        )
    ]

    weak_topics = [
        item["topic"]
        for item in topic_report
        if item["percentage"] < 60
    ]

    if percentage >= 90:

        performance = "Excellent"

        feedback = (
            "Excellent understanding. Continue practising "
            "with challenging questions."
        )

    elif percentage >= 75:

        performance = "Very Good"

        feedback = (
            "You have a good understanding. Review the topics "
            "where you lost marks."
        )

    elif percentage >= 60:

        performance = "Good"

        feedback = (
            "Your fundamentals are developing. Revise weaker "
            "concepts and practise again."
        )

    elif percentage >= 40:

        performance = "Needs Improvement"

        feedback = (
            "Revise the important concepts and try another "
            "practice quiz."
        )

    else:

        performance = "More Practice Needed"

        feedback = (
            "Start by revisiting the study notes and "
            "understanding the basic concepts."
        )

    revision_suggestions = []

    for item in topic_report:

        if item["percentage"] < 60:

            revision_suggestions.append(
                f"Revise '{item['topic']}' and attempt "
                "practice questions on this topic."
            )

        elif item["percentage"] < 75:

            revision_suggestions.append(
                f"Strengthen '{item['topic']}' with "
                "another round of revision."
            )

    if not revision_suggestions:

        revision_suggestions.append(
            "Your topic-wise performance is strong. "
            "Try a harder quiz to test deeper understanding."
        )

    try:

        time_taken_seconds = max(
            0,
            int(time_taken_seconds),
        )

    except (TypeError, ValueError):

        time_taken_seconds = 0

    minutes, seconds = divmod(
        time_taken_seconds,
        60,
    )

    return {
        "score": correct_count,
        "total_questions": total_questions,
        "percentage": percentage,
        "correct": correct_count,
        "incorrect": incorrect_count,
        "unanswered": unanswered_count,
        "performance": performance,
        "feedback": feedback,
        "time_taken_seconds": time_taken_seconds,
        "time_taken_formatted": (
            f"{minutes} min {seconds} sec"
        ),
        "strengths": strengths,
        "weak_topics": weak_topics,
        "topic_report": topic_report,
        "revision_suggestions": revision_suggestions,
        "results": results,
    }


# ================================================================
# PUBLIC HELPERS
# ================================================================

def get_supported_question_types() -> list:
    """Return supported question types for the Streamlit UI."""

    return [
        {
            "id": question_type,
            "label": QUESTION_TYPE_LABELS[question_type],
        }
        for question_type in (
            "mcq",
            "true_false",
            "assertion_reason",
            "multiple_select",
            "fill_blank",
            "scenario",
        )
    ]
