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

ALLOWED_TAGS = set(bleach.sanitizer.ALLOWED_TAGS) | {
    "p", "h1", "h2", "h3", "h4",
    "ul", "ol", "li",
    "table", "thead", "tbody", "tr", "th", "td",
    "pre", "code", "hr", "blockquote", "br",
    "div", "span", "strong", "em",
}

ALLOWED_ATTRIBUTES = {
    **bleach.sanitizer.ALLOWED_ATTRIBUTES,
    "th": ["align"],
    "td": ["align"],
    "code": ["class"],
    "pre": ["class"],
    "div": ["class"],
    "span": ["class"],
}


# ============================================================
# FONT HELPERS
# ============================================================

def font_as_data_uri(font_path):
    """Embed a local font into the generated HTML."""

    if not font_path.is_file():
        raise FileNotFoundError(
            f"Font not found: {font_path}\n"
            "Check that the required Noto font files exist "
            "inside the project's fonts/ directory."
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
        raise ValueError(
            "The supplied visual must be a valid PNG image."
        )

    encoded = base64.b64encode(
        image_bytes
    ).decode("ascii")

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

    value = re.sub(
        r"<[^>]+>",
        " ",
        value,
    )

    value = re.sub(
        r"[*_`#]",
        "",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip().lower()


def remove_duplicate_title(notes_html, title):
    """
    Remove the first Markdown H1 if it duplicates
    the title already displayed on the cover.
    """

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

    heading_text = html.unescape(
        heading_text
    )

    if normalize_text(heading_text) == normalize_text(title):
        notes_html = (
            notes_html[:match.start()]
            + notes_html[match.end():]
        )

    return notes_html


# ============================================================
# FORMULA AND REACTION BOXES
# ============================================================

def convert_special_blockquotes(rendered_html):
    """
    Convert special Markdown blockquotes into styled boxes.

    Supported markers:
        > [FORMULA] ...
        > [REACTION] ...
    """

    pattern = re.compile(
        r"<blockquote>\s*<p>\s*"
        r"\[(FORMULA|REACTION)\]\s*"
        r"(.*?)"
        r"</p>\s*</blockquote>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    def replace_block(match):

        box_type = match.group(1).upper()
        content = match.group(2).strip()

        if box_type == "FORMULA":
            label = "IMPORTANT FORMULA"
            css_class = "formula-box"
        else:
            label = "CHEMICAL REACTION"
            css_class = "reaction-box"

        return f"""
            <div class="{css_class}">
                <div class="special-box-label">
                    {label}
                </div>
                <div class="special-box-content">
                    {content}
                </div>
            </div>
        """

    return pattern.sub(
        replace_block,
        rendered_html,
    )


# ============================================================
# MARKDOWN CONVERSION
# ============================================================

def markdown_to_safe_html(text):
    """Convert Markdown notes into sanitized HTML."""

    rendered = markdown.markdown(
        text,
        extensions=[
            "tables",
            "fenced_code",
            "sane_lists",
        ],
    )

    rendered = convert_special_blockquotes(
        rendered
    )

    return bleach.clean(
        rendered,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        strip=True,
    )


# ============================================================
# DIAGRAM MATCHING
# ============================================================

def get_diagram_keywords(diagram):
    """Collect keywords from the visual metadata."""

    values = [
        str(diagram.get("title", "")),
        str(diagram.get("related_heading", "")),
        str(diagram.get("description", "")),
        str(diagram.get("visual_type", "")),
    ]

    visible_labels = diagram.get(
        "visible_labels",
        [],
    )

    if isinstance(visible_labels, list):
        values.extend(
            str(label)
            for label in visible_labels
        )

    for node in diagram.get("nodes", []):
        if isinstance(node, dict):
            values.append(
                str(node.get("label", ""))
            )
            values.append(
                str(node.get("details", ""))
            )
        else:
            values.append(str(node))

    combined = " ".join(values).lower()

    words = re.findall(
        r"[a-zA-Z0-9]+",
        combined,
    )

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
    """
    Find the most relevant H2-H4 heading.

    Exact related_heading matches receive priority.
    Keyword matching is used only as a fallback.
    """

    related_heading = normalize_text(
        diagram.get("related_heading", "")
    )

    if related_heading:

        for index, heading in enumerate(headings):

            heading_text = normalize_text(
                heading.group(2)
            )

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

        heading_text = normalize_text(
            heading.group(2)
        )

        heading_words = set(
            re.findall(
                r"[a-zA-Z0-9]+",
                heading_text,
            )
        )

        score = len(
            keywords.intersection(heading_words)
        )

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
    """
    Create a visual section with a caption.

    The caption is outside the image. No title is drawn
    over the original video frame.
    """

    title = html.escape(
        str(
            diagram.get(
                "title",
                f"Video Visual {index}",
            )
        )
    )

    description = html.escape(
        str(diagram.get("description", "")).strip()
    )

    timestamp = html.escape(
        str(diagram.get("timestamp_label", "")).strip()
    )

    source = html.escape(
        str(
            diagram.get(
                "source",
                "Original video visual",
            )
        )
    )

    visible_labels = diagram.get(
        "visible_labels",
        [],
    )

    if not isinstance(visible_labels, list):
        visible_labels = []

    labels_html = ""

    if visible_labels:
        labels = ", ".join(
            html.escape(str(label))
            for label in visible_labels[:12]
        )

        labels_html = f"""
            <p class="visual-labels">
                <strong>Visible labels:</strong> {labels}
            </p>
        """

    timestamp_html = ""

    if timestamp:
        timestamp_html = f"""
            <span class="visual-timestamp">
                Video timestamp: {timestamp}
            </span>
        """

    description_html = ""

    if description:
        description_html = f"""
            <p class="visual-description">
                {description}
            </p>
        """

    image_uri = image_as_data_uri(
        diagram.get("image", b"")
    )

    return f"""
        <figure class="diagram">

            <img
                src="{image_uri}"
                alt="{title}"
            >

            <figcaption>

                <div class="visual-title">
                    {title}
                </div>

                {timestamp_html}

                {description_html}

                {labels_html}

                <p class="visual-source">
                    Source: {source}
                </p>

            </figcaption>

        </figure>
    """


# ============================================================
# INSERT VISUALS NEAR MATCHING TOPICS
# ============================================================

def insert_diagrams_near_headings(
    notes_html,
    diagrams,
):
    """
    Insert each visual near its matching topic.

    Placement priority:
    1. Exact related_heading match.
    2. Keyword-based heading match.
    3. Additional Visuals appendix.

    The visual is inserted after the first explanatory
    paragraph of the matching section.
    """

    if not diagrams:
        return notes_html, []

    heading_pattern = re.compile(
        r"<h([2-4])\b[^>]*>(.*?)</h\1>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    headings = list(
        heading_pattern.finditer(notes_html)
    )

    if not headings:
        return notes_html, list(range(len(diagrams)))

    placements = {}
    appendix = []

    for diagram_index, diagram in enumerate(diagrams):

        heading_index = find_matching_heading(
            diagram,
            headings,
        )

        if heading_index is None:
            appendix.append(diagram_index)
        else:
            placements.setdefault(
                heading_index,
                [],
            ).append(diagram_index)

    result = notes_html

    # Insert from bottom to top so earlier positions remain valid.
    for heading_index in sorted(
        placements.keys(),
        reverse=True,
    ):

        heading = headings[heading_index]

        if heading_index + 1 < len(headings):
            next_heading_start = headings[
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
                heading.end()
                + first_paragraph.end()
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
# APPENDIX
# ============================================================

def create_visual_appendix(
    diagrams,
    appendix_indices,
):
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

            <h2>Additional Video Visuals</h2>

            <p class="appendix-intro">
                These visuals were found in the video but
                could not be confidently matched to a specific
                heading in the notes.
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

    if language == "Hindi":
        document_language = "hi"
    else:
        document_language = "en"

    notes_html = markdown_to_safe_html(notes)

    notes_html = remove_duplicate_title(
        notes_html,
        title,
    )

    notes_html, appendix_indices = (
        insert_diagrams_near_headings(
            notes_html,
            diagram_images,
        )
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

        <style>

            /* ==========================================
               EMBEDDED FONTS
            ========================================== */

            @font-face {{
                font-family: "NotoSans";
                src: url("{fonts['latin_regular']}") format("truetype");
                font-weight: 400;
                font-style: normal;
            }}

            @font-face {{
                font-family: "NotoSans";
                src: url("{fonts['latin_bold']}") format("truetype");
                font-weight: 700;
                font-style: normal;
            }}

            @font-face {{
                font-family: "NotoDevanagari";
                src: url("{fonts['devanagari_regular']}") format("truetype");
                font-weight: 400;
                font-style: normal;
            }}

            @font-face {{
                font-family: "NotoDevanagari";
                src: url("{fonts['devanagari_bold']}") format("truetype");
                font-weight: 700;
                font-style: normal;
            }}


            /* ==========================================
               PAGE SETTINGS
            ========================================== */

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


            /* ==========================================
               NOTEBOOK PAPER
            ========================================== */

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


            /* ==========================================
               COVER
            ========================================== */

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


            /* ==========================================
               NOTES CONTENT
            ========================================== */

            main {{
                background: rgba(255, 255, 255, 0.82);
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
                padding: 5px 9px;
                margin: 16px 0 9px;
                border-left: 4px solid #3478BD;
                background: rgba(220, 235, 255, 0.94);
                border-radius: 4px;
                line-height: 1.35;
                break-after: avoid;
            }}

            h2 {{
                color: #218052;
                font-size: 12pt;
                font-weight: 700;
                margin: 15px 0 5px;
                padding: 2px 0;
                border-bottom: 1px solid #B5DEC5;
                line-height: 1.35;
                break-after: avoid;
            }}

            h3 {{
                color: #784BB0;
                font-size: 10.5pt;
                font-weight: 700;
                margin: 9px 0 3px;
                line-height: 1.35;
                break-after: avoid;
            }}

            h4 {{
                color: #B96A21;
                font-size: 9.8pt;
                font-weight: 700;
                margin: 7px 0 3px;
                line-height: 1.3;
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
                margin-bottom: 2px;
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


            /* ==========================================
               TABLES
            ========================================== */

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


            /* ==========================================
               IMPORTANT NOTES
            ========================================== */

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


            /* ==========================================
               FORMULA BOXES
            ========================================== */

            .formula-box {{
                margin: 11px 0;
                padding: 11px 14px;
                border: 1.5px solid #4C8ACB;
                border-left: 5px solid #286DB0;
                border-radius: 7px;
                background: #EAF4FF;
                break-inside: avoid;
            }}

            .formula-box .special-box-label {{
                color: #245A9A;
                font-size: 8pt;
                font-weight: 700;
                letter-spacing: 0.5px;
                margin-bottom: 5px;
            }}

            .formula-box .special-box-content {{
                color: #173F70;
                font-size: 11pt;
                font-weight: 700;
                line-height: 1.6;
            }}


            /* ==========================================
               CHEMICAL REACTION BOXES
            ========================================== */

            .reaction-box {{
                margin: 11px 0;
                padding: 11px 14px;
                border: 1.5px solid #D59B4A;
                border-left: 5px solid #C47A24;
                border-radius: 7px;
                background: #FFF5E7;
                break-inside: avoid;
            }}

            .reaction-box .special-box-label {{
                color: #A55D16;
                font-size: 8pt;
                font-weight: 700;
                letter-spacing: 0.5px;
                margin-bottom: 5px;
            }}

            .reaction-box .special-box-content {{
                color: #75420F;
                font-size: 11pt;
                font-weight: 700;
                line-height: 1.6;
            }}


            /* ==========================================
               CODE
            ========================================== */

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


            /* ==========================================
               VIDEO VISUALS
            ========================================== */

            .diagram {{
                width: 100%;
                margin: 12px auto 16px;
                padding: 8px;
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

            .visual-timestamp {{
                display: inline-block;
                color: #218052;
                font-size: 8pt;
                font-weight: 700;
                margin: 2px 0 4px;
            }}

            .visual-description {{
                color: #52657D;
                font-size: 8.3pt;
                margin: 3px 0;
            }}

            .visual-labels {{
                color: #52657D;
                font-size: 8pt;
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


            /* ==========================================
               PRINT SAFETY
            ========================================== */

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

            .cover {{
                break-inside: avoid;
            }}

            .formula-box,
            .reaction-box {{
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
            optional visual metadata.

    Returns:
        PDF document as bytes.
    """

    if not isinstance(notes, str) or not notes.strip():
        raise ValueError("Study notes are empty.")

    diagram_images = diagram_images or []

    if not isinstance(diagram_images, list):
        raise ValueError(
            "diagram_images must be a list."
        )

    for index, diagram in enumerate(
        diagram_images,
        start=1,
    ):

        if not isinstance(diagram, dict):
            raise ValueError(
                f"Visual {index} must be a dictionary."
            )

        image_bytes = diagram.get("image")

        if not isinstance(image_bytes, bytes):
            raise ValueError(
                f"Visual {index} must contain PNG image bytes."
            )

        if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(
                f"Visual {index} contains invalid PNG data."
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
            )

            # Wait for embedded fonts.
            page.evaluate(
                "() => document.fonts.ready"
            )

            # Wait for every embedded image.
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