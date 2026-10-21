
import base64
import html
import os
from pathlib import Path

import bleach
import markdown
from playwright.sync_api import sync_playwright


BASE_DIR = Path(__file__).resolve().parent
FONT_DIR = BASE_DIR / "fonts"

FONT_FILES = {
    "latin_regular": FONT_DIR / "NotoSans-Regular.ttf",
    "latin_bold": FONT_DIR / "NotoSans-Bold.ttf",
    "devanagari_regular": FONT_DIR / "NotoSansDevanagari-Regular.ttf",
    "devanagari_bold": FONT_DIR / "NotoSansDevanagari-Bold.ttf",
}

ALLOWED_TAGS = set(bleach.sanitizer.ALLOWED_TAGS) | {
    "p", "h1", "h2", "h3", "h4",
    "ul", "ol", "li", "table", "thead",
    "tbody", "tr", "th", "td", "pre",
    "code", "hr", "blockquote", "br",
    "div", "span", "strong", "em",
}

ALLOWED_ATTRIBUTES = {
    **bleach.sanitizer.ALLOWED_ATTRIBUTES,
    "th": ["align"],
    "td": ["align"],
    "code": ["class"],
    "pre": ["class"],
    "div": ["class"],
}


def font_as_data_uri(font_path):
    """Embed a local font in the HTML used to create the PDF."""
    if not font_path.is_file():
        raise FileNotFoundError(
            f"Font not found: {font_path}\n"
            "Download the required Noto Sans TTF files "
            "and place them in the project's fonts/ directory."
        )

    encoded = base64.b64encode(font_path.read_bytes()).decode("ascii")
    return f"data:font/ttf;base64,{encoded}"


def image_as_data_uri(image_bytes):
    """Convert PNG/JPEG bytes into an embeddable HTML image."""
    if not image_bytes:
        raise ValueError("A diagram image is empty.")

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def markdown_to_safe_html(text):
    """Convert Markdown notes into HTML while allowing common formatting."""
    rendered = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists"],
    )

    return bleach.clean(
        rendered,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        strip=True,
    )


def create_study_notes_pdf(
    notes,
    title="YouTube Video Study Notes",
    language="English",
    diagram_images=None,
):
    """
    Create a PDF containing study notes and rendered diagram images.

    diagram_images format:
        [
            {"title": "Process Flow", "image": png_bytes},
            {"title": "System Architecture", "image": png_bytes},
        ]

    Returns:
        PDF as bytes.
    """
    if not isinstance(notes, str) or not notes.strip():
        raise ValueError("Study notes are empty.")

    diagram_images = diagram_images or []

    # Validate all fonts before creating the document.
    fonts = {
        name: font_as_data_uri(path)
        for name, path in FONT_FILES.items()
    }

    safe_title = html.escape(str(title))
    safe_language = html.escape(str(language))
    notes_html = markdown_to_safe_html(notes)

    diagram_sections = []

    for index, diagram in enumerate(diagram_images, start=1):
        diagram_title = html.escape(
            str(diagram.get("title") or f"Diagram {index}")
        )
        image_uri = image_as_data_uri(diagram.get("image", b""))

        diagram_sections.append(
            f"""
            <section class="diagram">
                <h2>{diagram_title}</h2>
                <img src="{image_uri}" alt="{diagram_title}">
            </section>
            """
        )

    diagrams_html = "\n".join(diagram_sections)

    document_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">

        <style>
            @font-face {{
                font-family: "NotoSans";
                src: url("{fonts['latin_regular']}") format("truetype");
                font-weight: 400;
            }}

            @font-face {{
                font-family: "NotoSans";
                src: url("{fonts['latin_bold']}") format("truetype");
                font-weight: 700;
            }}

            @font-face {{
                font-family: "NotoDevanagari";
                src: url("{fonts['devanagari_regular']}") format("truetype");
                font-weight: 400;
            }}

            @font-face {{
                font-family: "NotoDevanagari";
                src: url("{fonts['devanagari_bold']}") format("truetype");
                font-weight: 700;
            }}

            @page {{
                size: A4;
                margin: 20mm 17mm 20mm 17mm;

                @bottom-center {{
                    content: "Page " counter(page) " of " counter(pages);
                    font-family: "NotoSans", "NotoDevanagari", sans-serif;
                    font-size: 8pt;
                    color: #666;
                }}
            }}

            body {{
                font-family: "NotoSans", "NotoDevanagari", sans-serif;
                font-size: 10.5pt;
                line-height: 1.65;
                color: #222;
                overflow-wrap: anywhere;
            }}

            h1, h2, h3, h4, strong, th {{
                font-family: "NotoSans", "NotoDevanagari", sans-serif;
                font-weight: 700;
            }}

            .cover {{
                text-align: center;
                padding: 25px 0;
                border-bottom: 2px solid #345995;
                margin-bottom: 24px;
            }}

            .cover h1 {{
                color: #234b83;
                font-size: 23pt;
                line-height: 1.35;
            }}

            .subtitle {{
                color: #666;
                font-size: 9pt;
            }}

            h1, h2, h3 {{
                color: #234b83;
                line-height: 1.4;
                break-after: avoid;
            }}

            h2 {{
                margin-top: 22px;
                border-bottom: 1px solid #d6dfeb;
                padding-bottom: 4px;
            }}

            h3 {{
                margin-top: 16px;
            }}

            p, li, td, th, blockquote {{
                overflow-wrap: anywhere;
                word-wrap: break-word;
            }}

            ul, ol {{
                padding-left: 23px;
            }}

            li {{
                margin-bottom: 5px;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 14px 0;
                font-size: 9pt;
                break-inside: auto;
            }}

            tr {{
                break-inside: avoid;
            }}

            th, td {{
                border: 1px solid #ccd4df;
                padding: 6px;
                text-align: left;
                vertical-align: top;
            }}

            th {{
                background: #edf2f8;
            }}

            pre, code {{
                font-family: "NotoSans", "NotoDevanagari", monospace;
                white-space: pre-wrap;
                overflow-wrap: anywhere;
            }}

            pre {{
                background: #f4f6f8;
                padding: 10px;
                border-radius: 4px;
            }}

            blockquote {{
                border-left: 3px solid #8aa8ce;
                padding-left: 12px;
                color: #444;
            }}

            .diagram {{
                margin-top: 24px;
                break-inside: avoid;
            }}

            .diagram img {{
                display: block;
                max-width: 100%;
                max-height: 225mm;
                width: auto;
                height: auto;
                object-fit: contain;
                margin: 12px auto;
            }}

            .section-label {{
                color: #555;
                font-size: 9pt;
            }}
        </style>
    </head>

    <body>
        <header class="cover">
            <h1>{safe_title}</h1>
            <p class="subtitle">Language: {safe_language}</p>
            <p class="subtitle">Generated from your YouTube learning session</p>
        </header>

        <main>
            {notes_html}

            {(
                '<h1>Visual Diagrams</h1>' + diagrams_html
                if diagram_sections else ''
            )}
        </main>
    </body>
    </html>
    """

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(
            viewport={"width": 1240, "height": 1754},
            device_scale_factor=1,
        )

        try:
            page.set_content(document_html, wait_until="load")
            page.evaluate("() => document.fonts.ready")

            pdf_bytes = page.pdf(
                format="A4",
                print_background=True,
                prefer_css_page_size=True,
                display_header_footer=False,
                margin={
                    "top": "20mm",
                    "right": "17mm",
                    "bottom": "20mm",
                    "left": "17mm",
                },
            )
        finally:
            browser.close()

    if not pdf_bytes.startswith(b"%PDF"):
        raise RuntimeError("The PDF renderer did not return a valid PDF.")

    return pdf_bytes
