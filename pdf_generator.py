
import base64
import html
import re
import unicodedata
from pathlib import Path

import bleach
import markdown

from playwright.sync_api import sync_playwright


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
FONT_DIR = BASE_DIR / "fonts"

FONT_FILES = {
    "latin_regular": FONT_DIR / "NotoSans-Regular.ttf",
    "latin_bold": FONT_DIR / "NotoSans-Bold.ttf",
    "devanagari_regular": FONT_DIR / "NotoSansDevanagari-Regular.ttf",
    "devanagari_bold": FONT_DIR / "NotoSansDevanagari-Bold.ttf",
}

HANDWRITING_FONT_STACK = (
    '"Segoe Print", "Comic Sans MS", "Chalkboard", '
    '"NotoSans", "NotoDevanagari", sans-serif'
)

MATHJAX_URL = (
    "https://cdn.jsdelivr.net/npm/mathjax@3/"
    "es5/tex-chtml.js"
)

ALLOWED_TAGS = set(bleach.sanitizer.ALLOWED_TAGS) | {
    "p", "h1", "h2", "h3", "h4",
    "ul", "ol", "li",
    "table", "thead", "tbody", "tr", "th", "td",
    "pre", "code", "hr", "blockquote", "br",
    "div", "span", "strong", "em",
    "sup", "sub",
    "figure", "figcaption", "img",
}

ALLOWED_ATTRIBUTES = {
    **bleach.sanitizer.ALLOWED_ATTRIBUTES,
    "th": ["align"],
    "td": ["align"],
    "code": ["class"],
    "pre": ["class"],
    "div": ["class"],
    "span": ["class"],
    "figure": ["class"],
    "figcaption": ["class"],
    "img": ["src", "alt", "class"],
}


# ============================================================
# FONT HELPERS
# ============================================================

def font_as_data_uri(font_path):
    """Embed a local font into generated HTML."""

    if not font_path.is_file():
        raise FileNotFoundError(
            f"Font not found: {font_path}\n"
            "Check the required Noto font files inside fonts/."
        )

    encoded = base64.b64encode(
        font_path.read_bytes()
    ).decode("ascii")

    return f"data:font/ttf;base64,{encoded}"


# ============================================================
# IMAGE HELPERS
# ============================================================

def image_as_data_uri(image_bytes):
    """Convert valid PNG bytes into an embeddable HTML image."""

    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("A diagram image is empty or invalid.")

    if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("The supplied visual must be a valid PNG.")

    encoded = base64.b64encode(image_bytes).decode("ascii")

    return f"data:image/png;base64,{encoded}"


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(value):
    """Normalize text for reliable heading comparison."""

    value = unicodedata.normalize(
        "NFKC",
        str(value or ""),
    )

    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[*_`#]", "", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip().lower()


def remove_duplicate_title(notes_html, title):
    """Remove the first H1 if it duplicates the cover title."""

    pattern = re.compile(
        r"<h1\b[^>]*>(.*?)</h1>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    match = pattern.search(notes_html)

    if not match:
        return notes_html

    heading_text = re.sub(
        r"<[^>]+>",
        " ",
        match.group(1),
    )

    heading_text = html.unescape(heading_text)

    if normalize_text(heading_text) == normalize_text(title):
        notes_html = (
            notes_html[:match.start()]
            + notes_html[match.end():]
        )

    return notes_html


# ============================================================
# MATHEMATICAL DELIMITER PROTECTION
# ============================================================

def protect_math_delimiters(text):
    """
    Protect LaTeX delimiters before Markdown conversion.

    This prevents Markdown from interpreting LaTeX delimiters
    as ordinary formatting.
    """

    replacements = {
        r"\[": "MATHDISPLAYOPENPLACEHOLDER",
        r"\]": "MATHDISPLAYCLOSEPLACEHOLDER",
        r"\(": "MATHINLINEOPENPLACEHOLDER",
        r"\)": "MATHINLINECLOSEPLACEHOLDER",
    }

    for original, placeholder in replacements.items():
        text = text.replace(original, placeholder)

    return text


def restore_math_delimiters(rendered_html):
    """Restore protected LaTeX delimiters after Markdown."""

    replacements = {
        "MATHDISPLAYOPENPLACEHOLDER": r"\[",
        "MATHDISPLAYCLOSEPLACEHOLDER": r"\]",
        "MATHINLINEOPENPLACEHOLDER": r"\(",
        "MATHINLINECLOSEPLACEHOLDER": r"\)",
    }

    for placeholder, original in replacements.items():
        rendered_html = rendered_html.replace(
            placeholder,
            original,
        )

    return rendered_html


# ============================================================
# SPECIAL EDUCATIONAL BLOCKS
# ============================================================

def convert_special_blockquotes(rendered_html):
    """Convert labelled Markdown blockquotes into styled boxes."""

    block_pattern = re.compile(
        r"<blockquote\b[^>]*>(.*?)</blockquote>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    block_types = {
        "FORMULA": (
            "formula-box",
            "IMPORTANT MATHEMATICAL FORMULA",
        ),
        "REACTION": (
            "reaction-box",
            "CHEMICAL REACTION",
        ),
        "ADDITIONAL KNOWLEDGE": (
            "additional-knowledge-box",
            "ADDITIONAL KNOWLEDGE",
        ),
        "NEW ILLUSTRATIVE EXAMPLE": (
            "new-example-box",
            "NEW ILLUSTRATIVE EXAMPLE",
        ),
        "VIDEO EXAMPLE": (
            "video-example-box",
            "EXAMPLE FROM THE VIDEO",
        ),
        "WORKED EXAMPLE": (
            "worked-example-box",
            "WORKED EXAMPLE",
        ),
        "STEP-BY-STEP EXAMPLE": (
            "worked-example-box",
            "STEP-BY-STEP EXAMPLE",
        ),
    }

    def replace_block(match):
        inner_html = match.group(1)

        marker = re.search(
            r"\[\s*(FORMULA|REACTION|ADDITIONAL\s+KNOWLEDGE|"
            r"NEW\s+ILLUSTRATIVE\s+EXAMPLE|VIDEO\s+EXAMPLE|"
            r"WORKED\s+EXAMPLE|STEP-BY-STEP\s+EXAMPLE)\s*\]",
            inner_html,
            flags=re.IGNORECASE,
        )

        if not marker:
            return match.group(0)

        marker_name = re.sub(
            r"\s+",
            " ",
            marker.group(1).upper(),
        )

        css_class, label = block_types[marker_name]

        content = (
            inner_html[:marker.start()]
            + inner_html[marker.end():]
        )

        content = re.sub(
            r"^\s*(?:<p\b[^>]*>)?\s*"
            r"(?:<br\s*/?>\s*)?",
            "",
            content,
            count=1,
            flags=re.IGNORECASE,
        )

        content = re.sub(
            r"</p>\s*<p\b[^>]*>",
            "<br>",
            content,
            flags=re.IGNORECASE,
        )

        content = re.sub(
            r"</?p\b[^>]*>",
            "",
            content,
            flags=re.IGNORECASE,
        )

        content = content.strip()

        return f"""
            <div class="{css_class}">
                <div class="special-box-label">{label}</div>
                <div class="special-box-content">{content}</div>
            </div>
        """

    return block_pattern.sub(replace_block, rendered_html)


# ============================================================
# AUTOMATIC SECTION CLASSIFICATION
# ============================================================

def classify_educational_headings(rendered_html):
    """Apply styling classes to educational headings."""

    heading_pattern = re.compile(
        r"<h([1-4])\b([^>]*)>(.*?)</h\1>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    def replace_heading(match):
        level = match.group(1)
        attributes = match.group(2) or ""
        content = match.group(3)

        plain_text = html.unescape(
            re.sub(r"<[^>]+>", " ", content)
        )

        normalized = normalize_text(plain_text)

        heading_class = None

        if "additional knowledge" in normalized:
            heading_class = "additional-knowledge-heading"

        elif (
            "new illustrative example" in normalized
            or "new example" in normalized
        ):
            heading_class = "new-example-heading"

        elif "video example" in normalized:
            heading_class = "video-example-heading"

        elif (
            "worked example" in normalized
            or "step by step example" in normalized
            or "numerical example" in normalized
        ):
            heading_class = "worked-example-heading"

        if not heading_class:
            return match.group(0)

        attributes = re.sub(
            r'\sclass="[^"]*"',
            "",
            attributes,
            flags=re.IGNORECASE,
        )

        return (
            f'<h{level}{attributes} class="{heading_class}">'
            f"{content}</h{level}>"
        )

    return heading_pattern.sub(
        replace_heading,
        rendered_html,
    )


# ============================================================
# MARKDOWN CONVERSION
# ============================================================

def markdown_to_safe_html(text):
    """Convert Markdown notes into safe HTML."""

    protected_text = protect_math_delimiters(text)

    rendered = markdown.markdown(
        protected_text,
        extensions=[
            "tables",
            "fenced_code",
            "sane_lists",
        ],
    )

    rendered = restore_math_delimiters(rendered)
    rendered = convert_special_blockquotes(rendered)
    rendered = classify_educational_headings(rendered)

    return bleach.clean(
        rendered,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        protocols={"http", "https", "data"},
        strip=True,
    )


# ============================================================
# DIAGRAM MATCHING
# ============================================================

def get_diagram_keywords(diagram):
    """Collect keywords from diagram metadata."""

    values = [
        str(diagram.get("title", "")),
        str(diagram.get("related_heading", "")),
        str(diagram.get("description", "")),
        str(diagram.get("visual_type", "")),
        str(diagram.get("diagram_type", "")),
    ]

    visible_labels = diagram.get("visible_labels", [])

    if isinstance(visible_labels, list):
        values.extend(str(label) for label in visible_labels)

    for node in diagram.get("nodes", []):
        if isinstance(node, dict):
            values.append(str(node.get("label", "")))
            values.append(str(node.get("details", "")))
        else:
            values.append(str(node))

    combined = " ".join(values).lower()
    words = re.findall(r"[a-zA-Z0-9]+", combined)

    ignored = {
        "the", "and", "for", "with", "from",
        "into", "diagram", "concept", "main",
        "system", "basic", "using", "this",
        "that", "which", "their", "these",
        "details", "component", "example",
        "video", "visual", "frame", "original",
    }

    return {
        word
        for word in words
        if len(word) > 2 and word not in ignored
    }


def find_matching_heading(diagram, headings):
    """Find the most relevant H2-H4 heading."""

    related_heading = normalize_text(
        diagram.get("related_heading", "")
    )

    if related_heading:
        for index, heading in enumerate(headings):
            heading_text = normalize_text(heading.group(2))

            if (
                heading_text == related_heading
                or heading_text in related_heading
                or related_heading in heading_text
            ):
                return index

    keywords = get_diagram_keywords(diagram)

    best_heading_index = None
    best_score = 0

    for heading_index, heading in enumerate(headings):
        heading_text = normalize_text(heading.group(2))

        heading_words = set(
            re.findall(r"[a-zA-Z0-9]+", heading_text)
        )

        score = len(keywords.intersection(heading_words))

        if score > best_score:
            best_score = score
            best_heading_index = heading_index

    if best_score >= 1:
        return best_heading_index

    return None


# ============================================================
# DIAGRAM HTML
# ============================================================

def create_diagram_section(diagram, index):
    """Create a diagram section with a caption."""

    title = html.escape(
        str(diagram.get("title", f"Concept Diagram {index}"))
    )

    description = html.escape(
        str(diagram.get("description", "")).strip()
    )

    source = html.escape(
        str(
            diagram.get(
                "source",
                "Generated educational diagram",
            )
        )
    )

    image_uri = image_as_data_uri(
        diagram.get("image", b"")
    )

    return f"""
        <figure class="diagram">
            <img src="{image_uri}" alt="{title}">
            <figcaption>
                <div class="visual-title">{title}</div>
                <p class="visual-description">{description}</p>
                <p class="visual-source">{source}</p>
            </figcaption>
        </figure>
    """


# ============================================================
# INSERT DIAGRAMS NEAR MATCHING TOPICS
# ============================================================

def insert_diagrams_near_headings(notes_html, diagrams):
    """Insert diagrams near matching topic headings."""

    if not diagrams:
        return notes_html, []

    heading_pattern = re.compile(
        r"<h([2-4])\b[^>]*>(.*?)</h\1>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    initial_headings = list(
        heading_pattern.finditer(notes_html)
    )

    if not initial_headings:
        return notes_html, list(range(len(diagrams)))

    placements = {}
    appendix = []

    for diagram_index, diagram in enumerate(diagrams):
        heading_index = find_matching_heading(
            diagram,
            initial_headings,
        )

        if heading_index is None:
            appendix.append(diagram_index)
        else:
            placements.setdefault(
                heading_index,
                [],
            ).append(diagram_index)

    result = notes_html

    for heading_index in sorted(placements.keys(), reverse=True):
        current_headings = list(
            heading_pattern.finditer(result)
        )

        if heading_index >= len(current_headings):
            appendix.extend(placements[heading_index])
            continue

        heading = current_headings[heading_index]

        if heading_index + 1 < len(current_headings):
            next_heading_start = current_headings[
                heading_index + 1
            ].start()
        else:
            next_heading_start = len(result)

        section_text = result[
            heading.end():next_heading_start
        ]

        first_paragraph = re.search(
            r"</p\s*>",
            section_text,
            flags=re.IGNORECASE,
        )

        if first_paragraph:
            insertion_point = (
                heading.end() + first_paragraph.end()
            )
        else:
            insertion_point = heading.end()

        diagram_html = "\n".join(
            create_diagram_section(
                diagrams[index],
                index + 1,
            )
            for index in placements[heading_index]
        )

        result = (
            result[:insertion_point]
            + "\n"
            + diagram_html
            + "\n"
            + result[insertion_point:]
        )

    return result, appendix


# ============================================================
# DIAGRAM APPENDIX
# ============================================================

def create_visual_appendix(diagrams, appendix_indices):
    """Create an appendix for unmatched diagrams."""

    if not appendix_indices:
        return ""

    sections = []

    for index in appendix_indices:
        sections.append(
            create_diagram_section(
                diagrams[index],
                index + 1,
            )
        )

    return f"""
        <section class="diagram-appendix">
            <h2>Additional Concept Diagrams</h2>

            <p class="appendix-intro">
                These diagrams provide additional explanations
                for concepts covered in the study notes.
            </p>

            {"".join(sections)}
        </section>
    """


# ============================================================
# DOCUMENT HTML
# ============================================================

def create_document_html(
    notes,
    title,
    language,
    diagram_images,
):
    """Build the complete notebook-style HTML document."""

    if not isinstance(notes, str) or not notes.strip():
        raise ValueError("Study notes are empty.")

    diagram_images = diagram_images or []

    fonts = {
        name: font_as_data_uri(path)
        for name, path in FONT_FILES.items()
    }

    safe_title = html.escape(str(title))
    safe_language = html.escape(str(language))

    document_language = (
        "hi" if language == "Hindi" else "en"
    )

    notes_html = markdown_to_safe_html(notes)

    notes_html = remove_duplicate_title(
        notes_html,
        title,
    )

    notes_html, appendix_indices = insert_diagrams_near_headings(
        notes_html,
        diagram_images,
    )

    appendix_html = create_visual_appendix(
        diagram_images,
        appendix_indices,
    )

    document_html = f"""
    <!DOCTYPE html>
    <html lang="{document_language}">
    <head>
        <meta charset="UTF-8">

        <meta
            name="viewport"
            content="width=device-width, initial-scale=1.0"
        >

        <script>
            window.MathJax = {{
                loader: {{
                    load: ['[tex]/mhchem']
                }},

                tex: {{
                    packages: {{
                        '[+]': ['mhchem']
                    }},

                    inlineMath: [
                        ['\\\\(', '\\\\)'],
                        ['$', '$']
                    ],

                    displayMath: [
                        ['\\\\[', '\\\\]'],
                        ['$$', '$$']
                    ],

                    processEscapes: true,
                    processEnvironments: true,
                    tags: 'ams'
                }},

                options: {{
                    skipHtmlTags: [
                        'script',
                        'noscript',
                        'style',
                        'textarea',
                        'pre',
                        'code'
                    ]
                }},

                chtml: {{
                    scale: 1.0,
                    matchFontHeight: false
                }}
            }};
        </script>

        <script
            id="MathJax-script"
            async
            src="{MATHJAX_URL}"
        ></script>

        <style>

            @font-face {{
                font-family: "NotoSans";
                src: url("{fonts['latin_regular']}")
                    format("truetype");
                font-weight: 400;
            }}

            @font-face {{
                font-family: "NotoSans";
                src: url("{fonts['latin_bold']}")
                    format("truetype");
                font-weight: 700;
            }}

            @font-face {{
                font-family: "NotoDevanagari";
                src: url("{fonts['devanagari_regular']}")
                    format("truetype");
                font-weight: 400;
            }}

            @font-face {{
                font-family: "NotoDevanagari";
                src: url("{fonts['devanagari_bold']}")
                    format("truetype");
                font-weight: 700;
            }}

            @page {{
                size: A4;
                margin: 12mm 14mm 17mm 19mm;
            }}

            * {{
                box-sizing: border-box;
            }}

            html {{
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}

            body {{
                margin: 0;
                font-family: {HANDWRITING_FONT_STACK};
                font-size: 10pt;
                line-height: 1.65;
                color: #263D62;
                overflow-wrap: anywhere;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}

            /* NOTEBOOK PAPER */

            .paper {{
                position: relative;
                min-height: 100%;
                padding: 8px 15px 25px 28px;

                background-color: #FFFFFF;

                background-image:
                    linear-gradient(
                        to right,
                        transparent 0,
                        transparent 39px,
                        rgba(231, 112, 112, 0.38) 40px,
                        transparent 41px
                    ),
                    repeating-linear-gradient(
                        to bottom,
                        #FFFFFF 0px,
                        #FFFFFF 27px,
                        #DCE9F7 28px,
                        #FFFFFF 29px
                    );

                background-size:
                    100% 100%,
                    100% 29px;

                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}

            /* COVER */

            .cover {{
                text-align: center;
                padding: 12px 10px 15px;
                margin-bottom: 15px;
                border-bottom: 2px solid #3768A5;
                background: rgba(255, 255, 255, 0.96);
                break-inside: avoid;
            }}

            .cover h1 {{
                font-size: 20pt;
                font-weight: 700;
                line-height: 1.35;
                color: #245A9A;
                margin: 0 0 7px;
            }}

            .subtitle {{
                font-size: 8.5pt;
                color: #687A93;
                line-height: 1.4;
                margin: 3px 0;
            }}

            /* NOTES */

            main {{
                background: rgba(255, 255, 255, 0.88);
                padding: 5px 8px;
            }}

            p {{
                margin: 3px 0 8px;
                color: #263D62;
            }}

            h1 {{
                color: #245A9A;
                font-size: 16pt;
                font-weight: 700;
                padding: 6px 10px;
                margin: 16px 0 10px;
                border-left: 5px solid #3478BD;
                background: #DCEBFA;
                border-radius: 4px;
                line-height: 1.4;
                break-after: avoid;
            }}

            h2 {{
                color: #218052;
                font-size: 12.5pt;
                font-weight: 700;
                margin: 16px 0 7px;
                padding: 4px 8px;
                border-left: 4px solid #32966A;
                border-bottom: 1px solid #B5DEC5;
                background: rgba(230, 247, 237, 0.95);
                border-radius: 3px;
                line-height: 1.4;
                break-after: avoid;
            }}

            h3 {{
                color: #784BB0;
                font-size: 11pt;
                font-weight: 700;
                margin: 11px 0 5px;
                padding: 3px 7px;
                background: rgba(243, 235, 255, 0.95);
                border-left: 3px solid #8055B5;
                border-radius: 3px;
                line-height: 1.4;
                break-after: avoid;
            }}

            h4 {{
                color: #B96A21;
                font-size: 10pt;
                font-weight: 700;
                margin: 8px 0 4px;
                line-height: 1.35;
                break-after: avoid;
            }}

            strong {{
                color: #D14B38;
                font-weight: 700;
            }}

            em {{
                color: #168A91;
            }}

            ul, ol {{
                padding-left: 22px;
                margin: 4px 0 9px;
            }}

            li {{
                margin-bottom: 3px;
                padding-left: 2px;
                color: #263D62;
            }}

            li::marker {{
                color: #2E78B8;
            }}

            hr {{
                border: none;
                border-top: 1px dashed #B7C9DD;
                margin: 13px 0;
            }}

            /* SECTION HEADING VARIANTS */

            .additional-knowledge-heading {{
                color: #176B68;
                background: #E2F5F1;
                border-left: 5px solid #23978B;
                border-bottom: 1px solid #B7E3DB;
            }}

            .new-example-heading {{
                color: #8B541A;
                background: #FFF1D9;
                border-left: 5px solid #E5A13C;
                border-bottom: 1px solid #F1D6A6;
            }}

            .video-example-heading {{
                color: #245A9A;
                background: #E7F0FC;
                border-left: 5px solid #4787C7;
                border-bottom: 1px solid #C5D9F0;
            }}

            .worked-example-heading {{
                color: #70449C;
                background: #F1E9FB;
                border-left: 5px solid #9163BC;
                border-bottom: 1px solid #D9C9EC;
            }}

            /* TABLES */

            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 8px 0 12px;
                font-size: 9pt;
                background: rgba(255, 255, 255, 0.97);
                break-inside: auto;
            }}

            thead {{
                display: table-header-group;
            }}

            tr {{
                break-inside: avoid;
            }}

            th {{
                color: #245A9A;
                background: #DCEBFA;
                font-weight: 700;
            }}

            th, td {{
                border: 1px solid #B9CADD;
                padding: 5px 6px;
                text-align: left;
                vertical-align: top;
                overflow-wrap: anywhere;
            }}

            td {{
                color: #263D62;
            }}

            /* GENERAL BLOCKQUOTES */

            blockquote {{
                margin: 10px 0;
                padding: 7px 12px;
                border-left: 4px solid #E49B3B;
                background: #FFF4DF;
                color: #80551F;
                border-radius: 4px;
                break-inside: avoid;
            }}

            blockquote p {{
                color: #80551F;
                margin: 2px 0;
            }}

            /* EDUCATIONAL BOX BASE */

            .formula-box,
            .reaction-box,
            .additional-knowledge-box,
            .new-example-box,
            .video-example-box,
            .worked-example-box {{
                margin: 13px 0;
                padding: 12px 15px;
                border-radius: 8px;
                break-inside: avoid;
                page-break-inside: avoid;
            }}

            .special-box-label {{
                font-size: 8pt;
                font-weight: 700;
                letter-spacing: 0.5px;
                margin-bottom: 7px;
            }}

            .special-box-content {{
                font-size: 9.5pt;
                line-height: 1.75;
                overflow-wrap: anywhere;
            }}

            .special-box-content p:last-child {{
                margin-bottom: 0;
            }}

            /* FORMULAS */

            .formula-box {{
                border: 1.5px solid #4C8ACB;
                border-left: 5px solid #286DB0;
                background: #EAF4FF;
            }}

            .formula-box .special-box-label {{
                color: #245A9A;
            }}

            .formula-box .special-box-content {{
                color: #173F70;
                font-size: 11pt;
                font-weight: 600;
            }}

            .formula-box mjx-container {{
                color: #173F70 !important;
            }}

            /* CHEMICAL REACTIONS */

            .reaction-box {{
                border: 1.5px solid #D59B4A;
                border-left: 5px solid #C47A24;
                background: #FFF5E7;
            }}

            .reaction-box .special-box-label {{
                color: #A55D16;
            }}

            .reaction-box .special-box-content {{
                color: #75420F;
                font-size: 11pt;
                font-weight: 600;
            }}

            .reaction-box mjx-container {{
                color: #75420F !important;
            }}

            /* ADDITIONAL KNOWLEDGE */

            .additional-knowledge-box {{
                border: 1px solid #A8DCD3;
                border-left: 5px solid #23978B;
                background: #EDFAF7;
            }}

            .additional-knowledge-box .special-box-label {{
                color: #176B68;
            }}

            .additional-knowledge-box .special-box-content {{
                color: #285E5A;
            }}

            /* NEW ILLUSTRATIVE EXAMPLES */

            .new-example-box {{
                border: 1px solid #F0D19C;
                border-left: 5px solid #E5A13C;
                background: #FFF8EA;
            }}

            .new-example-box .special-box-label {{
                color: #925A1D;
            }}

            .new-example-box .special-box-content {{
                color: #74532C;
            }}

            /* VIDEO EXAMPLES */

            .video-example-box {{
                border: 1px solid #C4D9F1;
                border-left: 5px solid #4787C7;
                background: #F0F6FD;
            }}

            .video-example-box .special-box-label {{
                color: #245A9A;
            }}

            .video-example-box .special-box-content {{
                color: #385C83;
            }}

            /* WORKED EXAMPLES */

            .worked-example-box {{
                border: 1px solid #D7C4EA;
                border-left: 5px solid #9163BC;
                background: #F7F1FC;
            }}

            .worked-example-box .special-box-label {{
                color: #70449C;
            }}

            .worked-example-box .special-box-content {{
                color: #57436D;
            }}

            /* MATHEMATICAL EXPRESSIONS */

            mjx-container {{
                font-size: 105% !important;
                max-width: 100%;
            }}

            mjx-container[display="true"] {{
                display: block;
                max-width: 100%;
                overflow-x: auto;
                overflow-y: hidden;
                margin: 12px 0 !important;
                padding: 6px 2px;
            }}

            mjx-container[display="false"] {{
                display: inline-block;
                max-width: 100%;
            }}

            mjx-container[jax="CHTML"] {{
                outline: none;
            }}

            /* CODE */

            pre, code {{
                font-family:
                    "NotoSans",
                    "NotoDevanagari",
                    monospace;
                white-space: pre-wrap;
                overflow-wrap: anywhere;
            }}

            pre {{
                background: #F0F4F8;
                color: #334B6B;
                border: 1px solid #D2DFEC;
                border-radius: 5px;
                padding: 9px;
                line-height: 1.45;
                font-size: 8.5pt;
            }}

            /* DIAGRAMS */

            .diagram {{
                width: 100%;
                margin: 13px auto 17px;
                padding: 9px;
                background: #FFFFFF;
                border: 1px solid #D7E0EA;
                border-radius: 7px;
                break-inside: avoid;
                page-break-inside: avoid;
            }}

            .diagram img {{
                display: block;
                width: 100%;
                height: auto;
                max-height: 165mm;
                object-fit: contain;
                margin: 0 auto;
                background: #FFFFFF;
            }}

            .diagram figcaption {{
                padding: 7px 5px 2px;
                border-top: 1px solid #E7EDF4;
                margin-top: 7px;
                background: #FFFFFF;
            }}

            .visual-title {{
                color: #245A9A;
                font-size: 9.5pt;
                font-weight: 700;
                margin-bottom: 3px;
            }}

            .visual-description {{
                color: #52657D;
                font-size: 8.3pt;
                margin: 3px 0;
            }}

            .visual-source {{
                color: #8995A5;
                font-size: 7.3pt;
                margin: 4px 0 0;
            }}

            .diagram-appendix {{
                margin-top: 20px;
            }}

            .appendix-intro {{
                color: #64748B;
                font-size: 9pt;
                margin-bottom: 12px;
            }}

            /* PRINT SAFETY */

            img {{
                max-width: 100%;
            }}

            p, li, blockquote {{
                orphans: 2;
                widows: 2;
            }}

            h1, h2, h3, h4 {{
                break-inside: avoid;
            }}

            .cover,
            .formula-box,
            .reaction-box,
            .additional-knowledge-box,
            .new-example-box,
            .video-example-box,
            .worked-example-box {{
                break-inside: avoid;
                page-break-inside: avoid;
            }}

        </style>
    </head>

    <body>
        <div class="paper">

            <header class="cover">
                <h1>{safe_title}</h1>

                <p class="subtitle">
                    Language: {safe_language}
                </p>

                <p class="subtitle">
                    Complete Video Study Notes
                </p>
            </header>

            <main>
                {notes_html}
                {appendix_html}
            </main>

        </div>
    </body>
    </html>
    """

    return document_html


# ============================================================
# MATHJAX RENDERING
# ============================================================

def render_mathjax(page):
    """
    Wait for MathJax and typeset all mathematical expressions.

    Returns True if MathJax loads and typesets successfully.
    """

    try:
        page.wait_for_function(
            """
            () => (
                window.MathJax &&
                window.MathJax.startup &&
                window.MathJax.startup.promise &&
                typeof window.MathJax.typesetPromise === "function"
            )
            """,
            timeout=30000,
        )

        page.evaluate(
            """
            async () => {
                await window.MathJax.startup.promise;

                await window.MathJax.typesetPromise();

                await document.fonts.ready;
            }
            """
        )

        result = page.evaluate(
            """
            () => ({
                loaded: !!window.MathJax,

                rendered: document.querySelectorAll(
                    'mjx-container'
                ).length,

                errors: document.querySelectorAll(
                    'mjx-merror'
                ).length
            })
            """
        )

        print("MathJax rendering result:", result)

        return (
            result["loaded"]
            and result["rendered"] > 0
            and result["errors"] == 0
        )

    except Exception as error:
        print(f"MathJax rendering failed: {error}")
        return False


# ============================================================
# CREATE PDF
# ============================================================

def create_study_notes_pdf(
    notes,
    title="Video Study Notes",
    language="English",
    diagram_images=None,
):
    """
    Generate the notebook-style study notes PDF.

    Parameters:
        notes:
            Markdown study notes.

        title:
            Document title.

        language:
            English, Hindi or Hinglish.

        diagram_images:
            List of dictionaries containing PNG bytes and
            optional diagram metadata.

    Returns:
        PDF document as bytes.
    """

    if not isinstance(notes, str) or not notes.strip():
        raise ValueError("Study notes are empty.")

    diagram_images = diagram_images or []

    if not isinstance(diagram_images, list):
        raise ValueError("diagram_images must be a list.")

    for index, diagram in enumerate(diagram_images, start=1):
        if not isinstance(diagram, dict):
            raise ValueError(
                f"Diagram {index} must be a dictionary."
            )

        image_bytes = diagram.get("image")

        if not isinstance(image_bytes, bytes):
            raise ValueError(
                f"Diagram {index} must contain PNG image bytes."
            )

        if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(
                f"Diagram {index} contains invalid PNG data."
            )

    document_html = create_document_html(
        notes=notes,
        title=title,
        language=language,
        diagram_images=diagram_images,
    )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()

        try:
            page = browser.new_page(
                viewport={
                    "width": 1240,
                    "height": 1754,
                },
                device_scale_factor=1,
            )

            page.set_content(
                document_html,
                wait_until="load",
                timeout=45000,
            )

            page.evaluate(
                "() => document.fonts.ready"
            )

            page.evaluate(
                """async () => {
                    const images = Array.from(document.images);

                    await Promise.all(
                        images.map((img) => {
                            if (img.complete) {
                                return Promise.resolve();
                            }

                            return new Promise((resolve) => {
                                img.onload = resolve;
                                img.onerror = resolve;
                            });
                        })
                    );
                }"""
            )

            math_rendered = render_mathjax(page)

            page.evaluate(
                "() => new Promise(resolve => "
                "requestAnimationFrame(() => "
                "requestAnimationFrame(resolve)))"
            )

            if not math_rendered and re.search(
                r"\\\(|\\\[|\$\$",
                notes,
            ):
                print(
                    "WARNING: MathJax could not successfully "
                    "render the mathematical expressions. "
                    "Check internet access and the LaTeX syntax."
                )

            pdf_bytes = page.pdf(
                format="A4",
                print_background=True,
                prefer_css_page_size=True,
                display_header_footer=True,
                header_template="<div></div>",
                footer_template="""
                    <div style="
                        width: 100%;
                        text-align: center;
                        font-family: Arial, sans-serif;
                        font-size: 8px;
                        color: #60738D;
                    ">
                        Page <span class="pageNumber"></span>
                        of <span class="totalPages"></span>
                    </div>
                """,
                margin={
                    "top": "12mm",
                    "right": "14mm",
                    "bottom": "17mm",
                    "left": "19mm",
                },
            )

        finally:
            browser.close()

    if not pdf_bytes.startswith(b"%PDF"):
        raise RuntimeError(
            "The PDF renderer did not return a valid PDF."
        )

    return pdf_bytes
