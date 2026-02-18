"""
StructureDetector - Intelligent document structure analysis.

Detects:
- Chapter boundaries
- Section hierarchy
- Paragraph groupings
- Content type classification (abstract, introduction, conclusion, etc.)
- Reading order
"""

import re
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, List, Literal
from pathlib import Path

logger = logging.getLogger(__name__)


# Known semantic section labels
SEMANTIC_LABELS = {
    r"abstract": "abstract",
    r"introduction|overview": "introduction",
    r"background|related work|literature review": "background",
    r"method(ology|s)?|approach|proposed": "methodology",
    r"experiment(s|al)?|evaluation|results?": "results",
    r"discussion": "discussion",
    r"conclusion(s)?|summary": "conclusion",
    r"acknowledge?ments?": "acknowledgements",
    r"references?|bibliography|works cited": "references",
    r"appendix|appendices": "appendix",
    r"table of contents|contents": "toc",
}


@dataclass
class Section:
    section_id: str
    title: str
    level: int
    content: str
    paragraphs: list[dict] = field(default_factory=list)
    subsections: list["Section"] = field(default_factory=list)
    semantic_label: Optional[str] = None
    page_start: int = 0
    page_end: int = 0
    word_count: int = 0

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "level": self.level,
            "content": self.content,
            "paragraphs": self.paragraphs,
            "semantic_label": self.semantic_label,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "word_count": self.word_count,
        }


class StructureDetector:
    """
    Intelligent hierarchical document structure detection.

    Algorithm:
    1. Classify each element as heading (H1-H6) or body text
    2. Calibrate heading levels by font size statistics
    3. Build chapter/section tree
    4. Assign semantic labels (Introduction, Methods, etc.)
    5. Group paragraphs under their parent sections
    """

    def __init__(
        self,
        min_heading_length: int = 2,
        max_heading_length: int = 200,
        min_paragraph_words: int = 3,
    ):
        self.min_heading_length = min_heading_length
        self.max_heading_length = max_heading_length
        self.min_paragraph_words = min_paragraph_words

    def detect(self, raw_content: dict) -> dict:
        """
        Detect structure from raw parsed content.

        Args:
            raw_content: Raw dict from any parser

        Returns:
            dict with 'chapters', 'sections', 'paragraphs' keys
        """
        elements = raw_content.get("raw_elements", [])

        if not elements:
            return {"chapters": [], "sections": [], "paragraphs": []}

        # Calibrate heading detection
        elements = self._calibrate_headings(elements)

        # Build flat section list and paragraphs
        sections, paragraphs = self._build_structure(elements)

        # Detect chapters (top-level H1 sections)
        chapters = self._extract_chapters(sections)

        # Add semantic labels
        for section in sections:
            section["semantic_label"] = self._classify_semantic(section["title"])

        logger.debug(
            f"Structure: {len(chapters)} chapters, "
            f"{len(sections)} sections, "
            f"{len(paragraphs)} paragraphs"
        )

        return {
            "chapters": chapters,
            "sections": [s for s in sections],
            "paragraphs": paragraphs,
        }

    def _calibrate_headings(self, elements: list[dict]) -> list[dict]:
        """
        Re-calibrate heading levels based on font size distribution.

        Some PDFs encode all text with the same style; we use font size
        statistics to determine the true heading hierarchy.
        """
        # Gather font sizes
        font_sizes = [
            el.get("font_size", 0)
            for el in elements
            if el.get("font_size", 0) > 0
        ]

        if not font_sizes:
            return elements

        # Compute percentiles
        sorted_sizes = sorted(set(font_sizes), reverse=True)
        body_size = self._estimate_body_size(font_sizes)

        # Map large font sizes to heading levels
        heading_size_map = {}
        level = 1
        for size in sorted_sizes:
            if size > body_size * 1.1 and level <= 6:
                heading_size_map[size] = level
                level += 1

        # Re-assign heading types based on calibrated sizes
        updated = []
        for el in elements:
            el_copy = dict(el)
            font_size = el.get("font_size", 0)
            flags = el.get("flags", 0)
            is_bold = bool(flags & 16) or el.get("bold", False)  # fitz bold flag = bit 4

            # If already classified as heading, trust it
            if "heading_" in el.get("type", ""):
                updated.append(el_copy)
                continue

            # Classify by font size
            if font_size in heading_size_map:
                lvl = heading_size_map[font_size]
                text = el.get("content", "").strip()

                # Heuristics for valid headings
                words = text.split()
                is_mostly_numeric = all(re.match(r'^[\d\.\-\:]+$', w) for w in words)
                is_equation = "=" in text or "\u03c0" in text or len(re.findall(r'[+\-*/^]', text)) > 1

                if (
                    self.min_heading_length <= len(text) <= self.max_heading_length
                    and not text.endswith(".")
                    and not text.endswith(",")
                    and not is_mostly_numeric
                    and not is_equation
                    and (is_bold or font_size > body_size * 1.25)
                ):
                    el_copy["type"] = f"heading_{lvl}"
                    el_copy["heading_level"] = lvl

            updated.append(el_copy)

        return updated

    def _estimate_body_size(self, font_sizes: list[float]) -> float:
        """Estimate the dominant body text font size (mode)."""
        size_counts = Counter(round(s) for s in font_sizes)
        if not size_counts:
            return 12.0
        return size_counts.most_common(1)[0][0]

    def _build_structure(
        self, elements: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """Build sections and paragraphs from elements."""
        sections = []
        paragraphs = []
        current_section = None
        section_counter = 0
        para_counter = 0
        current_page = 0

        # Buffer for merging consecutive lines into a single paragraph
        text_buffer = []
        buffer_section_id = None
        buffer_page = 0
        buffer_block_id = None

        def flush_buffer():
            nonlocal para_counter
            if not text_buffer:
                return

            para_text = " ".join(text_buffer).strip()
            text_buffer.clear()

            words = para_text.split()
            if len(words) < self.min_paragraph_words:
                return

            para_counter += 1
            sec_id = buffer_section_id
            para = {
                "para_id": f"para_{para_counter:05d}",
                "text": para_text,
                "section_id": sec_id if sec_id else "root",
                "page": buffer_page,
                "type": "paragraph",
                "word_count": len(words),
            }
            paragraphs.append(para)
            if sec_id:
                # Find the section and append
                for s in sections:
                    if s["section_id"] == sec_id:
                        s["content"] += (" " if s["content"] else "") + para_text
                        s["paragraphs"].append(para["para_id"])
                        break

        for el in elements:
            el_type = el.get("type", "")
            text = el.get("content", "").strip()
            page = el.get("page", current_page)
            block_id = el.get("block_id")
            current_page = page

            if not text:
                continue

            if "heading_" in el_type:
                flush_buffer()
                level = el.get("heading_level") or int(el_type.split("_")[-1])

                # Close previous section
                if current_section:
                    current_section["page_end"] = page
                    current_section["word_count"] = len(
                        current_section["content"].split()
                    )

                section_counter += 1
                current_section = {
                    "section_id": f"sec_{section_counter:04d}",
                    "title": text,
                    "level": level,
                    "content": "",
                    "paragraphs": [],
                    "semantic_label": None,
                    "page_start": page,
                    "page_end": page,
                    "word_count": 0,
                }
                sections.append(current_section)
                buffer_section_id = current_section["section_id"]

            elif el_type in ("paragraph", "text", "ocr_text", "list_item"):
                # If block_id changed or page changed, flush buffer
                if (block_id != buffer_block_id or page != buffer_page) and text_buffer:
                    flush_buffer()

                if not text_buffer:
                    buffer_section_id = current_section["section_id"] if current_section else None
                    buffer_page = page
                    buffer_block_id = block_id

                text_buffer.append(text)

        flush_buffer() # Final flush

        # Close last section
        if current_section:
            current_section["page_end"] = current_page
            current_section["word_count"] = len(current_section["content"].split())

        return sections, paragraphs

    def _extract_chapters(self, sections: list[dict]) -> list[dict]:
        """Extract top-level H1 sections as chapters."""
        chapters = []
        chapter_counter = 0
        current_chapter = None

        for section in sections:
            if section["level"] == 1:
                chapter_counter += 1
                current_chapter = {
                    "chapter_id": f"chap_{chapter_counter:03d}",
                    "chapter_number": chapter_counter,
                    "title": section["title"],
                    "section_id": section["section_id"],
                    "semantic_label": section.get("semantic_label"),
                    "page_start": section["page_start"],
                    "page_end": section["page_end"],
                    "subsection_ids": [],
                    "word_count": section["word_count"],
                }
                chapters.append(current_chapter)
            elif current_chapter and section["level"] > 1:
                current_chapter["subsection_ids"].append(section["section_id"])
                current_chapter["page_end"] = section["page_end"]
                current_chapter["word_count"] += section["word_count"]

        return chapters

    def _classify_semantic(self, title: str) -> Optional[str]:
        """Classify a section title into a semantic category."""
        title_lower = title.lower().strip()
        for pattern, label in SEMANTIC_LABELS.items():
            if re.search(pattern, title_lower):
                return label
        return None


# Utility: detect input types from a list of file paths (files or directories).
def detect_input_types(paths: List[str]) -> Literal['text-only', 'image-only', 'combined']:
    """
    Inspect given paths (files or directories) non-recursively and classify as:
      - 'text-only'  : only text-like files present (.txt, .md, .pdf, .docx, .html)
      - 'image-only' : only image file extensions present (.jpg, .jpeg, .png, .tiff, .bmp)
      - 'combined'   : mixture of both or unknown files

    Directories are scanned non-recursively; their immediate children are inspected.
    """
    text_exts = {'.txt', '.md', '.markdown', '.pdf', '.docx', '.doc', '.html', '.htm', '.epub'}
    image_exts = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif'}

    found_text = False
    found_image = False

    for p in paths:
        pth = Path(p)
        if pth.is_dir():
            try:
                for child in pth.iterdir():
                    if child.is_file():
                        ext = child.suffix.lower()
                        if ext in text_exts:
                            found_text = True
                        elif ext in image_exts:
                            found_image = True
            except Exception:
                # If directory can't be accessed, treat as unknown -> combined
                return 'combined'
        else:
            ext = pth.suffix.lower()
            if ext in text_exts:
                found_text = True
            elif ext in image_exts:
                found_image = True
            else:
                # Unknown file type -> treat as combined to be safe
                return 'combined'

        if found_text and found_image:
            return 'combined'

    if found_text and not found_image:
        return 'text-only'
    if found_image and not found_text:
        return 'image-only'
    return 'combined'
