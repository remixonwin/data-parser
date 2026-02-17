"""
PDF Parser - Robust PDF content extraction.

Uses pdfminer for text with layout, PyMuPDF (fitz) for images,
and pytesseract for OCR on scanned pages.
"""

import logging
import io
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PDFParser:
    """
    Multi-strategy PDF parser.

    Strategy order:
    1. PyMuPDF (fitz) - fastest, good layout
    2. pdfminer.six - fallback for complex layouts
    3. OCR via pytesseract - for scanned/image-only PDFs
    """

    def __init__(self, enable_ocr: bool = True):
        self.enable_ocr = enable_ocr

    def parse(self, file_path: Path) -> dict:
        """Parse PDF and return raw content dict."""
        result = {
            "title": "",
            "authors": [],
            "created_at": "",
            "page_count": 0,
            "metadata": {},
            "raw_text": "",
            "raw_elements": [],  # List of {type, content, page, bbox}
            "raw_images": [],    # List of {data, page, bbox, format}
            "footnotes": [],
            "references": [],
        }

        try:
            import fitz  # PyMuPDF
            return self._parse_with_fitz(file_path, result)
        except ImportError:
            logger.warning("PyMuPDF not installed, trying pdfminer...")

        try:
            return self._parse_with_pdfminer(file_path, result)
        except ImportError:
            logger.warning("pdfminer not installed, trying basic pypdf...")

        try:
            return self._parse_with_pypdf(file_path, result)
        except ImportError:
            raise RuntimeError(
                "No PDF library available. Install: pip install pymupdf pdfminer.six"
            )

    def scout(self, file_path: Path) -> dict:
        """Quickly extract metadata and potential title from the first page."""
        try:
            import fitz
            doc = fitz.open(str(file_path))
            title = doc.metadata.get("title", "").strip()
            
            # If metadata title is missing or generic, look at the first page
            if not title or len(title) < 4 or title.lower().endswith(".pdf"):
                page = doc[0]
                # Get text blocks sorted by size/position
                blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
                
                # Heuristic: the largest text on the first page is likely the title
                best_span = None
                max_size = 0
                
                for b in blocks:
                    if b["type"] == 0:  # Text block
                        for l in b["lines"]:
                            for s in l["spans"]:
                                txt = s["text"].strip()
                                # Skip very short or numeric-only strings
                                if len(txt) > 3 and not txt.isdigit():
                                    if s["size"] > max_size:
                                        max_size = s["size"]
                                        best_span = txt
                
                if best_span:
                    title = best_span

            doc.close()
            return {"title": title or file_path.stem}
        except Exception as e:
            logger.debug(f"Title scouting failed: {e}")
            return {"title": file_path.stem}

    def _parse_with_fitz(self, file_path: Path, result: dict) -> dict:
        """Parse using PyMuPDF - best quality."""
        import fitz

        doc = fitz.open(str(file_path))
        result["page_count"] = doc.page_count

        # Extract metadata
        meta = doc.metadata
        result["title"] = meta.get("title", "")
        result["authors"] = [meta.get("author", "")] if meta.get("author") else []
        result["created_at"] = meta.get("creationDate", "")
        result["metadata"] = {
            "subject": meta.get("subject", ""),
            "creator": meta.get("creator", ""),
            "producer": meta.get("producer", ""),
            "format": meta.get("format", ""),
            "encrypted": doc.is_encrypted,
        }

        # Try TOC for chapter structure hints
        toc = doc.get_toc()
        if toc:
            result["metadata"]["toc"] = [
                {"level": item[0], "title": item[1], "page": item[2]}
                for item in toc
            ]

        all_text_parts = []
        raw_elements = []
        raw_images = []

        for page_num, page in enumerate(doc):
            page_idx = page_num + 1

            # Extract text blocks with position data
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

            for block in blocks:
                if block["type"] == 0:  # Text block
                    block_text_parts = []
                    for line in block["lines"]:
                        # Merge spans in a line
                        line_text = "".join(s["text"] for s in line["spans"]).strip()
                        if not line_text:
                            continue
                            
                        # Use the most prominent span's properties for the line
                        main_span = max(line["spans"], key=lambda s: len(s["text"]))
                        
                        element = {
                            "type": "text",
                            "content": line_text,
                            "page": page_idx,
                            "bbox": line["bbox"],
                            "font": main_span["font"],
                            "font_size": main_span["size"],
                            "flags": main_span["flags"],
                            "color": main_span["color"],
                            "origin": main_span["origin"],
                            "block_id": block.get("number", 0),
                        }
                        raw_elements.append(element)
                        all_text_parts.append(line_text)

                elif block["type"] == 1:  # Image block
                    # Handled below by page.get_images() for better coverage
                    pass

            # Improved image extraction: catch all images on the page
            try:
                xrefs = page.get_images(full=True)
                for img_info in xrefs:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    if base_image:
                        raw_images.append({
                            "image_bytes": base_image["image"],
                            "format": base_image["ext"],
                            "width": base_image["width"],
                            "height": base_image["height"],
                            "page": page_idx,
                            "bbox": [0, 0, 0, 0], # Bbox not easily available for get_images() without more work
                            "colorspace": base_image.get("colorspace", ""),
                            "xref": xref,
                        })
            except Exception as e:
                logger.debug(f"Image extraction error on page {page_idx}: {e}")

            # Check if page is scanned (very little text extracted)
            page_text = " ".join(
                s["content"] for s in raw_elements if s["page"] == page_idx
            )
            if self.enable_ocr and len(page_text.split()) < 10:
                ocr_text = self._ocr_page(page)
                if ocr_text:
                    raw_elements.append({
                        "type": "ocr_text",
                        "content": ocr_text,
                        "page": page_idx,
                        "bbox": [0, 0, page.rect.width, page.rect.height],
                        "font": "OCR",
                        "font_size": 0,
                        "flags": 0,
                        "color": 0,
                        "origin": (0, 0),
                    })
                    all_text_parts.append(ocr_text)

        doc.close()

        result["raw_text"] = "\n".join(all_text_parts)
        result["raw_elements"] = raw_elements

        # De-duplicate images by xref
        seen_xrefs = set()
        unique_images = []
        for img in raw_images:
            xref = img.get("xref", id(img))
            if xref not in seen_xrefs:
                seen_xrefs.add(xref)
                unique_images.append(img)
        result["raw_images"] = unique_images

        # Extract references (last pages often have References section)
        result["references"] = self._extract_references(raw_elements)

        return result

    def _parse_with_pdfminer(self, file_path: Path, result: dict) -> dict:
        """Fallback: parse using pdfminer.six."""
        from pdfminer.high_level import extract_pages, extract_text
        from pdfminer.layout import LTTextBox, LTFigure, LTPage

        text = extract_text(str(file_path))
        result["raw_text"] = text

        elements = []
        page_count = 0
        for page_layout in extract_pages(str(file_path)):
            page_count += 1
            for element in page_layout:
                if isinstance(element, LTTextBox):
                    elements.append({
                        "type": "text",
                        "content": element.get_text().strip(),
                        "page": page_count,
                        "bbox": list(element.bbox),
                        "font": "",
                        "font_size": 0,
                        "flags": 0,
                        "color": 0,
                        "origin": (0, 0),
                    })

        result["raw_elements"] = elements
        result["page_count"] = page_count
        return result

    def _parse_with_pypdf(self, file_path: Path, result: dict) -> dict:
        """Last resort: pypdf basic extraction."""
        import pypdf

        reader = pypdf.PdfReader(str(file_path))
        result["page_count"] = len(reader.pages)

        info = reader.metadata
        if info:
            result["title"] = info.get("/Title", "")
            result["authors"] = [info.get("/Author", "")] if info.get("/Author") else []

        texts = []
        elements = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            texts.append(text)
            if text.strip():
                elements.append({
                    "type": "text",
                    "content": text,
                    "page": i + 1,
                    "bbox": [0, 0, 600, 800],
                    "font": "",
                    "font_size": 0,
                    "flags": 0,
                    "color": 0,
                    "origin": (0, 0),
                })

        result["raw_text"] = "\n".join(texts)
        result["raw_elements"] = elements
        return result

    def _ocr_page(self, page) -> str:
        """OCR a fitz page using pytesseract."""
        try:
            import pytesseract
            from PIL import Image as PILImage
            import numpy as np

            mat = page.get_pixmap(dpi=300)
            img_array = np.frombuffer(mat.samples, dtype=np.uint8).reshape(
                mat.height, mat.width, mat.n
            )
            pil_img = PILImage.fromarray(img_array)
            return pytesseract.image_to_string(pil_img, config="--oem 3 --psm 6")
        except Exception as e:
            logger.debug(f"OCR failed: {e}")
            return ""

    def _extract_references(self, elements: list[dict]) -> list[dict]:
        """Heuristically extract bibliography/references section."""
        refs = []
        in_refs = False
        ref_pattern = re.compile(r"^\[(\d+)\]|^(\d+)\.|^\((\w+,?\s*\d{4})\)")

        for el in elements:
            text = el.get("content", "")
            if re.match(r"^(References|Bibliography|Works Cited)\s*$", text, re.I):
                in_refs = True
                continue
            if in_refs:
                if ref_pattern.match(text):
                    refs.append({"text": text, "page": el["page"]})
                elif refs and len(text) < 20 and text.isupper():
                    # Hit a new section header - stop
                    break

        return refs
