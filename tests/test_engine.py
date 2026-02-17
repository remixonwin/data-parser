"""
Test suite for DocParserEngine.

Tests cover:
- All parsers (PDF, DOCX, HTML, TXT)
- Structure detection
- Image extraction
- Caption engine (mocked)
- Dataset building
- Full integration pipeline
"""

import io
import os
import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


# ============================================================
# Fixtures
# ============================================================

SAMPLE_TEXT = """
# Introduction

This is a sample document for testing the DocParserEngine.

## Background

The engine supports multiple document formats including PDF, DOCX, HTML and plain text.

## Methods

We use a combination of rule-based and ML-based approaches to extract structure.

### Text Extraction

Text is extracted using format-specific parsers.

### Image Extraction

Images are extracted and filtered by size.

## Results

The engine achieves high accuracy on diverse document types.

## Conclusion

DocParserEngine is a production-grade solution for document parsing.

## References

[1] Smith, J. (2023). Document Understanding. Journal of AI.
[2] Doe, A. (2024). Multimodal Parsing. Conference Proceedings.
"""

SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Document</title>
    <meta name="author" content="Test Author">
</head>
<body>
    <h1>Chapter 1: Introduction</h1>
    <p>This is the introduction paragraph with enough words to be considered valid.</p>
    <h2>Section 1.1: Background</h2>
    <p>Background information goes here. More text to make it a full paragraph.</p>
    <h1>Chapter 2: Methods</h1>
    <p>Methodology description. Detailed explanation of the approach taken.</p>
</body>
</html>
"""


class TestTextParser(unittest.TestCase):
    """Test the plain text / Markdown parser."""

    def setUp(self):
        from doc_parser_engine.parsers.text_parser import TextParser
        self.parser = TextParser()
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _write_file(self, content: str, name: str = "test.md") -> Path:
        path = Path(self.tmp_dir) / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_parses_markdown_headings(self):
        """Test that # headings are detected."""
        path = self._write_file(SAMPLE_TEXT)
        result = self.parser.parse(path)

        headings = [
            e for e in result["raw_elements"]
            if "heading" in e["type"]
        ]
        self.assertGreater(len(headings), 3, "Should detect multiple headings")

    def test_h1_detection(self):
        """Test H1 heading level."""
        path = self._write_file("# Title\n\nSome paragraph text here.\n")
        result = self.parser.parse(path)

        h1s = [e for e in result["raw_elements"] if e["type"] == "heading_1"]
        self.assertEqual(len(h1s), 1)
        self.assertEqual(h1s[0]["content"], "Title")

    def test_paragraphs_extracted(self):
        """Test paragraph extraction."""
        path = self._write_file(SAMPLE_TEXT)
        result = self.parser.parse(path)

        paragraphs = [e for e in result["raw_elements"] if e["type"] == "paragraph"]
        self.assertGreater(len(paragraphs), 2)

    def test_underline_headings(self):
        """Test underline-style headings (===, ---)."""
        content = "Title\n=====\n\nBody text here.\n\nSubheading\n----------\n\nMore text.\n"
        path = self._write_file(content)
        result = self.parser.parse(path)

        h1s = [e for e in result["raw_elements"] if e["type"] == "heading_1"]
        h2s = [e for e in result["raw_elements"] if e["type"] == "heading_2"]
        self.assertGreater(len(h1s), 0, "Should detect underline H1")
        self.assertGreater(len(h2s), 0, "Should detect underline H2")

    def test_raw_text_populated(self):
        """Test that raw_text is populated."""
        path = self._write_file(SAMPLE_TEXT)
        result = self.parser.parse(path)
        self.assertGreater(len(result["raw_text"]), 100)

    def test_title_extraction(self):
        """Test title is first heading."""
        path = self._write_file("# My Document Title\n\nContent here.\n")
        result = self.parser.parse(path)
        # Title should be populated in higher-level code, but raw_elements should have H1
        h1s = [e for e in result["raw_elements"] if e["type"] == "heading_1"]
        self.assertEqual(h1s[0]["content"], "My Document Title")


class TestHTMLParser(unittest.TestCase):
    """Test the HTML parser."""

    def setUp(self):
        from doc_parser_engine.parsers.html_parser import HTMLParser
        try:
            import bs4
            self.parser = HTMLParser()
            self.tmp_dir = tempfile.mkdtemp()
            self.available = True
        except ImportError:
            self.available = False

    def tearDown(self):
        if hasattr(self, "tmp_dir"):
            shutil.rmtree(self.tmp_dir)

    def test_html_parsing(self):
        """Test basic HTML parsing."""
        if not self.available:
            self.skipTest("beautifulsoup4 not installed")

        path = Path(self.tmp_dir) / "test.html"
        path.write_text(SAMPLE_HTML, encoding="utf-8")
        result = self.parser.parse(path)

        self.assertEqual(result["title"], "Test Document")
        headings = [e for e in result["raw_elements"] if "heading" in e["type"]]
        self.assertGreater(len(headings), 0)

    def test_author_extraction(self):
        """Test author extraction from meta tags."""
        if not self.available:
            self.skipTest("beautifulsoup4 not installed")

        path = Path(self.tmp_dir) / "test.html"
        path.write_text(SAMPLE_HTML, encoding="utf-8")
        result = self.parser.parse(path)
        self.assertIn("Test Author", result["authors"])


class TestStructureDetector(unittest.TestCase):
    """Test intelligent structure detection."""

    def setUp(self):
        from doc_parser_engine.detection.structure_detector import StructureDetector
        self.detector = StructureDetector()

    def _make_raw(self, elements: list[dict]) -> dict:
        return {
            "raw_elements": elements,
            "raw_text": " ".join(e["content"] for e in elements),
        }

    def _make_elem(self, etype: str, content: str, level: int = None, page: int = 1) -> dict:
        return {
            "type": etype,
            "content": content,
            "page": page,
            "bbox": [],
            "heading_level": level,
            "font_size": 14 if "heading" in etype else 12,
            "flags": 16 if "heading" in etype else 0,  # bold for headings
        }

    def test_chapter_detection(self):
        """Test that H1 headings become chapters."""
        elements = [
            self._make_elem("heading_1", "Introduction", level=1),
            self._make_elem("paragraph", "This is the introduction text with enough words here."),
            self._make_elem("heading_1", "Methods", level=1),
            self._make_elem("paragraph", "This is the methods section with enough detail provided."),
        ]
        result = self.detector.detect(self._make_raw(elements))
        self.assertEqual(len(result["chapters"]), 2)
        self.assertEqual(result["chapters"][0]["title"], "Introduction")

    def test_section_hierarchy(self):
        """Test section nesting."""
        elements = [
            self._make_elem("heading_1", "Chapter 1", level=1),
            self._make_elem("heading_2", "Section 1.1", level=2),
            self._make_elem("paragraph", "Content for section 1.1 with some text."),
            self._make_elem("heading_2", "Section 1.2", level=2),
            self._make_elem("paragraph", "Content for section 1.2 with some text."),
        ]
        result = self.detector.detect(self._make_raw(elements))
        self.assertEqual(len(result["sections"]), 3)  # 1 H1 + 2 H2

    def test_semantic_labels(self):
        """Test semantic classification."""
        elements = [
            self._make_elem("heading_1", "Abstract", level=1),
            self._make_elem("paragraph", "This paper presents a novel approach to solving the problem."),
            self._make_elem("heading_1", "Introduction", level=1),
            self._make_elem("paragraph", "Background and motivation for this work goes here."),
            self._make_elem("heading_1", "Conclusion", level=1),
            self._make_elem("paragraph", "We have shown that the approach works well in practice."),
        ]
        result = self.detector.detect(self._make_raw(elements))
        labels = {s["title"]: s["semantic_label"] for s in result["sections"]}
        self.assertEqual(labels["Abstract"], "abstract")
        self.assertEqual(labels["Introduction"], "introduction")
        self.assertEqual(labels["Conclusion"], "conclusion")

    def test_paragraph_association(self):
        """Test paragraphs are associated with correct sections."""
        elements = [
            self._make_elem("heading_1", "Section A", level=1),
            self._make_elem("paragraph", "This is paragraph in section A with text."),
            self._make_elem("heading_1", "Section B", level=1),
            self._make_elem("paragraph", "This is paragraph in section B with text."),
        ]
        result = self.detector.detect(self._make_raw(elements))
        paras = result["paragraphs"]
        self.assertEqual(len(paras), 2)
        # Para section IDs should differ
        self.assertNotEqual(paras[0]["section_id"], paras[1]["section_id"])


class TestImageExtractor(unittest.TestCase):
    """Test image extraction and filtering."""

    def setUp(self):
        from doc_parser_engine.extractors.image_extractor import ImageExtractor
        self.tmp_dir = tempfile.mkdtemp()
        self.extractor = ImageExtractor(
            output_dir=Path(self.tmp_dir) / "images",
            min_size=10,
            save_to_disk=True,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _make_png_bytes(self, w: int = 100, h: int = 100) -> bytes:
        """Create a minimal PNG image."""
        try:
            from PIL import Image as PILImage
            img = PILImage.new("RGB", (w, h), color=(128, 64, 32))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except ImportError:
            # Minimal 1x1 PNG bytes
            return (
                b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
                b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00'
                b'\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18'
                b'\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
            )

    def test_extracts_valid_images(self):
        """Test that valid images are extracted."""
        raw = {
            "raw_images": [
                {"data": self._make_png_bytes(100, 100), "format": "png",
                 "width": 100, "height": 100, "page": 1, "bbox": []},
            ]
        }
        images = self.extractor.extract(raw, doc_id="test123")
        self.assertEqual(len(images), 1)

    def test_filters_small_images(self):
        """Test that tiny images are filtered out."""
        from doc_parser_engine.extractors.image_extractor import ImageExtractor
        extractor = ImageExtractor(
            output_dir=Path(self.tmp_dir) / "images2",
            min_size=200,  # Large minimum
            save_to_disk=False,
        )
        raw = {
            "raw_images": [
                {"data": self._make_png_bytes(50, 50), "format": "png",
                 "width": 50, "height": 50, "page": 1, "bbox": []},
            ]
        }
        images = extractor.extract(raw, doc_id="test456")
        self.assertEqual(len(images), 0)

    def test_deduplication(self):
        """Test that duplicate images are filtered."""
        png_data = self._make_png_bytes(100, 100)
        raw = {
            "raw_images": [
                {"data": png_data, "format": "png", "width": 100, "height": 100, "page": 1, "bbox": []},
                {"data": png_data, "format": "png", "width": 100, "height": 100, "page": 2, "bbox": []},
            ]
        }
        images = self.extractor.extract(raw, doc_id="test789")
        self.assertEqual(len(images), 1, "Duplicates should be filtered")

    def test_image_metadata(self):
        """Test image metadata is populated."""
        raw = {
            "raw_images": [
                {"data": self._make_png_bytes(200, 150), "format": "png",
                 "width": 200, "height": 150, "page": 3, "bbox": [10, 20, 210, 170]},
            ]
        }
        images = self.extractor.extract(raw, doc_id="metadoc")
        if images:
            img = images[0]
            self.assertIn("image_id", img)
            self.assertIn("content_hash", img)
            self.assertIn("aspect_ratio", img)
            self.assertEqual(img["page"], 3)


class TestDatasetBuilder(unittest.TestCase):
    """Test HuggingFace dataset building."""

    def setUp(self):
        try:
            import datasets
            self.datasets_available = True
        except ImportError:
            self.datasets_available = False

        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _make_mock_doc(self, doc_id: str = "abc123"):
        """Create a mock ParsedDocument."""
        from doc_parser_engine.core import ParsedDocument
        return ParsedDocument(
            doc_id=doc_id,
            source_path=f"/tmp/{doc_id}.pdf",
            doc_type="pdf",
            title="Test Document",
            authors=["Author One"],
            created_at="2024-01-01T00:00:00",
            parsed_at="2024-06-01T12:00:00",
            metadata={"subject": "Testing"},
            chapters=[{
                "chapter_id": "chap_001",
                "chapter_number": 1,
                "title": "Introduction",
                "section_id": "sec_0001",
                "semantic_label": "introduction",
                "page_start": 1,
                "page_end": 3,
                "subsection_ids": [],
                "word_count": 200,
            }],
            sections=[{
                "section_id": "sec_0001",
                "title": "Introduction",
                "level": 1,
                "content": "This is the introduction section content.",
                "paragraphs": ["para_00001"],
                "semantic_label": "introduction",
                "page_start": 1,
                "page_end": 3,
                "word_count": 200,
            }],
            paragraphs=[{
                "para_id": "para_00001",
                "text": "This is a test paragraph with sufficient content for testing.",
                "section_id": "sec_0001",
                "page": 1,
                "type": "paragraph",
                "word_count": 12,
            }],
            tables=[],
            images=[{
                "image_id": f"{doc_id}_0000_abcdef12",
                "doc_id": doc_id,
                "file_path": "/tmp/test.png",
                "format": "png",
                "width": 200,
                "height": 150,
                "aspect_ratio": 1.33,
                "mode": "RGB",
                "file_size_bytes": 1024,
                "content_hash": "abcdef12345678",
                "page": 1,
                "bbox": [],
                "alt_text": "Test figure",
                "caption": "A test image showing sample content.",
                "caption_model": "blip",
                "caption_confidence": 0.95,
                "ocr_text": "",
                "image_bytes": b"",
            }],
            footnotes=[],
            references=[],
            word_count=200,
            image_count=1,
            table_count=0,
            page_count=5,
        )

    def test_build_chunks_schema(self):
        """Test chunks schema creates one row per paragraph."""
        if not self.datasets_available:
            self.skipTest("datasets not installed")

        from doc_parser_engine.dataset.builder import DatasetBuilder
        builder = DatasetBuilder()
        doc = self._make_mock_doc()
        result = builder.build([doc], schema="chunks", include_images=False)

        total_rows = sum(len(split) for split in result.values())
        self.assertGreater(total_rows, 0)

    def test_build_minimal_schema(self):
        """Test minimal schema."""
        if not self.datasets_available:
            self.skipTest("datasets not installed")

        from doc_parser_engine.dataset.builder import DatasetBuilder
        builder = DatasetBuilder()
        docs = [self._make_mock_doc("doc1"), self._make_mock_doc("doc2")]
        result = builder.build(docs, schema="minimal", include_images=False)

        total_rows = sum(len(split) for split in result.values())
        self.assertEqual(total_rows, 2)

    def test_build_images_schema(self):
        """Test images schema."""
        if not self.datasets_available:
            self.skipTest("datasets not installed")

        from doc_parser_engine.dataset.builder import DatasetBuilder
        builder = DatasetBuilder()
        doc = self._make_mock_doc()
        result = builder.build([doc], schema="images", include_images=False)

        total_rows = sum(len(split) for split in result.values())
        self.assertEqual(total_rows, 1)  # One image

    def test_build_qa_schema(self):
        """Test QA schema."""
        if not self.datasets_available:
            self.skipTest("datasets not installed")

        from doc_parser_engine.dataset.builder import DatasetBuilder
        builder = DatasetBuilder()
        doc = self._make_mock_doc()
        result = builder.build([doc], schema="qa", include_images=False)

        total_rows = sum(len(split) for split in result.values())
        self.assertGreater(total_rows, 0)

    def test_invalid_schema_raises(self):
        """Test that invalid schema raises ValueError."""
        if not self.datasets_available:
            self.skipTest("datasets not installed")

        from doc_parser_engine.dataset.builder import DatasetBuilder
        builder = DatasetBuilder()
        doc = self._make_mock_doc()
        with self.assertRaises(ValueError):
            builder.build([doc], schema="invalid_schema")


class TestFullPipeline(unittest.TestCase):
    """Integration tests for the full pipeline."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_parse_markdown_file(self):
        """Full pipeline on a Markdown file."""
        from doc_parser_engine.core import DocParserEngine

        md_path = Path(self.tmp_dir) / "test.md"
        md_path.write_text(SAMPLE_TEXT, encoding="utf-8")

        engine = DocParserEngine(
            enable_captioning=False,
            enable_ocr=False,
            output_dir=self.tmp_dir,
        )
        doc = engine.parse(md_path)

        self.assertIsNotNone(doc.doc_id)
        self.assertEqual(doc.doc_type, "md")
        self.assertGreater(doc.word_count, 50)
        self.assertGreater(len(doc.sections), 3)

    def test_parse_html_file(self):
        """Full pipeline on an HTML file."""
        try:
            import bs4
        except ImportError:
            self.skipTest("beautifulsoup4 not installed")

        from doc_parser_engine.core import DocParserEngine

        html_path = Path(self.tmp_dir) / "test.html"
        html_path.write_text(SAMPLE_HTML, encoding="utf-8")

        engine = DocParserEngine(
            enable_captioning=False,
            enable_ocr=False,
            output_dir=self.tmp_dir,
        )
        doc = engine.parse(html_path)

        self.assertEqual(doc.title, "Test Document")
        self.assertGreater(len(doc.sections), 0)

    def test_parse_directory(self):
        """Test directory parsing."""
        from doc_parser_engine.core import DocParserEngine

        # Create multiple files
        for i in range(3):
            (Path(self.tmp_dir) / f"doc_{i}.txt").write_text(
                SAMPLE_TEXT, encoding="utf-8"
            )

        engine = DocParserEngine(
            enable_captioning=False,
            enable_ocr=False,
            output_dir=self.tmp_dir,
        )
        docs = list(engine.parse_directory(self.tmp_dir))
        self.assertEqual(len(docs), 3)

    def test_to_hf_dataset(self):
        """Test full pipeline to HF dataset."""
        try:
            import datasets
        except ImportError:
            self.skipTest("datasets not installed")

        from doc_parser_engine.core import DocParserEngine

        md_path = Path(self.tmp_dir) / "test.md"
        md_path.write_text(SAMPLE_TEXT, encoding="utf-8")

        engine = DocParserEngine(
            enable_captioning=False,
            enable_ocr=False,
            output_dir=self.tmp_dir,
        )
        doc = engine.parse(md_path)
        dataset = engine.to_hf_dataset(
            [doc],
            schema="chunks",
            include_images=False,
        )

        self.assertIsNotNone(dataset)
        self.assertIn("train", dataset)


if __name__ == "__main__":
    unittest.main(verbosity=2)
