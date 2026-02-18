"""
DocParserEngine - Core orchestrator for document parsing pipeline.

Supports: PDF, DOCX, HTML, TXT, EPUB, Markdown
Output: HuggingFace-compatible datasets with text + image content
"""

import os
import uuid
import logging
import hashlib
from pathlib import Path
from typing import Optional, Union, Iterator, List
from dataclasses import dataclass, field, asdict
from datetime import datetime

from .parsers.pdf_parser import PDFParser
from .parsers.docx_parser import DOCXParser
from .parsers.html_parser import HTMLParser
from .parsers.text_parser import TextParser
from .parsers.epub_parser import EPUBParser
from .extractors.image_extractor import ImageExtractor
from .extractors.table_extractor import TableExtractor
from .captioning.caption_engine import CaptionEngine
from .detection.structure_detector import StructureDetector, detect_input_types
from .dataset.builder import DatasetBuilder
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    """Fully parsed document representation."""
    doc_id: str
    source_path: str
    doc_type: str
    title: str
    authors: list[str]
    created_at: str
    parsed_at: str
    metadata: dict

    # Content
    chapters: list[dict] = field(default_factory=list)
    sections: list[dict] = field(default_factory=list)
    paragraphs: list[dict] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    footnotes: list[dict] = field(default_factory=list)
    references: list[dict] = field(default_factory=list)

    # Stats
    word_count: int = 0
    image_count: int = 0
    table_count: int = 0
    page_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class DocParserEngine:
    """
    Production-grade document parsing engine.

    Features:
    - Multi-format support (PDF, DOCX, HTML, TXT, EPUB, MD)
    - Intelligent chapter/section detection
    - Image extraction + AI captioning
    - Table extraction
    - HuggingFace Datasets export
    - Batch processing
    - Streaming support for large corpora

    Usage:
        engine = DocParserEngine()
        doc = engine.parse("document.pdf")
        dataset = engine.to_hf_dataset([doc])
        dataset.push_to_hub("username/my-dataset")
    """

    SUPPORTED_FORMATS = {
        ".pdf": PDFParser,
        ".docx": DOCXParser,
        ".doc": DOCXParser,
        ".html": HTMLParser,
        ".htm": HTMLParser,
        ".txt": TextParser,
        ".md": TextParser,
        ".markdown": TextParser,
        ".epub": EPUBParser,
    }

    def __init__(
        self,
        caption_model: str = "Salesforce/blip2-opt-2.7b",
        enable_captioning: bool = True,
        enable_ocr: bool = True,
        enable_table_extraction: bool = True,
        enable_ocr_on_images: bool = True,
        output_dir: Optional[str] = None,
        device: str = "auto",
        batch_size: int = 8,
        image_min_size: int = 100,
        llm_api_base: Optional[str] = None,
        llm_model: str = "gpt-4o",
        force_local_caption: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize DocParserEngine.

        Args:
            caption_model: HuggingFace model ID for image captioning
            enable_captioning: Whether to generate AI captions for images
            enable_ocr: Whether to apply OCR on image-heavy documents
            enable_table_extraction: Whether to extract tables as structured data
            output_dir: Directory for extracted images and artifacts
            device: "auto", "cpu", "cuda", or "mps"
            batch_size: Batch size for caption generation
            image_min_size: Minimum pixel size to extract images (width or height)
            verbose: Enable verbose logging
        """
        self.enable_captioning = enable_captioning
        self.enable_ocr = enable_ocr
        self.enable_table_extraction = enable_table_extraction
        self.batch_size = batch_size
        self.image_min_size = image_min_size
        self.enable_ocr_on_images = enable_ocr_on_images
        self.output_dir = Path(output_dir) if output_dir else Path("./parsed_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            level=level,
        )

        # Resolve device
        self.device = self._resolve_device(device)
        logger.info(f"DocParserEngine initialized | device={self.device}")

        # LLM API setup
        self.llm_api_base = llm_api_base
        self.llm_model = llm_model
        self.force_local_caption = force_local_caption
        self._llm_client = LLMClient(api_base=llm_api_base, default_model=llm_model) if llm_api_base else None

        # Initialize subsystems (lazy-loaded for performance)
        self._caption_engine: Optional[CaptionEngine] = None
        self._caption_model_id = caption_model
        self._structure_detector = StructureDetector()
        self._image_extractor = ImageExtractor(
            output_dir=self.output_dir / "images",
            min_size=image_min_size,
        )
        self._table_extractor = TableExtractor() if enable_table_extraction else None
        self._dataset_builder = DatasetBuilder()

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @property
    def caption_engine(self) -> "CaptionEngine":
        """Lazy-load caption engine."""
        if self._caption_engine is None and self.enable_captioning:
            # Pass force_local flag and llm_client so caption engine can decide whether to load HF models
            self._caption_engine = CaptionEngine(
                model_id=self._caption_model_id,
                device=self.device,
                batch_size=self.batch_size,
                enable_ocr_on_images=self.enable_ocr_on_images,
                llm_client=self._llm_client,
                force_local=self.force_local_caption,
            )
        return self._caption_engine

    def _get_parser(self, file_path: Path):
        """Get the appropriate parser for a file."""
        ext = file_path.suffix.lower()
        parser_cls = self.SUPPORTED_FORMATS.get(ext)
        if not parser_cls:
            raise ValueError(
                f"Unsupported file format: '{ext}'. "
                f"Supported formats: {list(self.SUPPORTED_FORMATS.keys())}"
            )
        return parser_cls(enable_ocr=self.enable_ocr)

    def scout_metadata(self, file_path: Union[str, Path]) -> dict:
        """
        Quickly extract metadata (like title) from a document without full parsing.
        Useful for intelligent naming before long tasks.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return {"title": file_path.stem}
        
        try:
            parser = self._get_parser(file_path)
            if hasattr(parser, "scout"):
                return parser.scout(file_path)
        except Exception as e:
            logger.debug(f"Metadata scouting failed: {e}")
            
        return {"title": file_path.stem}

    def parse(self, file_path: Union[str, Path]) -> ParsedDocument:
        """
        Parse a single document.

        Args:
            file_path: Path to the document

        Returns:
            ParsedDocument with all extracted content
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        logger.info(f"Parsing: {file_path.name}")

        # Document type detection for pipeline rules
        input_type = detect_input_types([str(file_path)])

        # Generate stable doc ID from file hash
        doc_id = self._compute_doc_id(file_path)

        # Parse raw content
        parser = self._get_parser(file_path)
        raw = parser.parse(file_path)

        # Detect document structure (chapters, sections)
        logger.debug("Running structure detection...")
        structure = self._structure_detector.detect(raw)

        # Extract images (but for text-only skip image extraction)
        images = []
        if input_type != 'text-only':
            logger.debug("Extracting images...")
            images = self._image_extractor.extract(raw, doc_id=doc_id)

        # Decide caption backend dynamically
        if input_type == 'text-only':
            # disable captioning and OCR
            should_caption = False
            should_ocr = False
        elif input_type == 'image-only':
            should_caption = True
            should_ocr = True
            # prefer remote API if available unless force_local_caption is True
            if self._llm_client and not self.force_local_caption:
                self._caption_model_id = 'api'
                logger.info(f"Using remote LLM API at {self.llm_api_base} with model {self.llm_model}; captioning backend set to 'api' (no large HF downloads)")
        else:  # combined
            should_caption = True
            should_ocr = True

        # Apply runtime overrides
        if not self.enable_captioning:
            should_caption = False
        if not self.enable_ocr:
            should_ocr = False

        # Generate captions
        if should_caption and images:
            logger.debug(f"Captioning {len(images)} images...")
            # ensure caption engine uses current model id and llm_client/force_local settings
            # reset caption engine if model id changed
            if self._caption_engine is not None and self._caption_engine.model_id != self._caption_model_id:
                self._caption_engine = None
            images = self.caption_engine.caption_batch(images)
        else:
            # ensure images have empty caption fields
            for img in images:
                img.setdefault('caption', '')
                img.setdefault('caption_model', '')
                img.setdefault('caption_confidence', 0.0)

        # For text-only, ensure images list empty
        if input_type == 'text-only':
            images = []

        # Extract tables
        tables = []
        if self._table_extractor and raw.get("raw_elements"):
            logger.debug("Extracting tables...")
            tables = self._table_extractor.extract(raw["raw_elements"])

        # Assemble parsed document
        doc = ParsedDocument(
            doc_id=doc_id,
            source_path=str(file_path.resolve()),
            doc_type=file_path.suffix.lower().lstrip("."),
            title=raw.get("title", file_path.stem),
            authors=raw.get("authors", []),
            created_at=raw.get("created_at", ""),
            parsed_at=datetime.utcnow().isoformat(),
            metadata=raw.get("metadata", {}),
            chapters=structure.get("chapters", []),
            sections=structure.get("sections", []),
            paragraphs=structure.get("paragraphs", []),
            tables=tables,
            images=images,
            footnotes=raw.get("footnotes", []),
            references=raw.get("references", []),
            word_count=self._count_words(structure.get("paragraphs", [])),
            image_count=len(images),
            table_count=len(tables),
            page_count=raw.get("page_count", 0),
        )

        logger.info(
            f"Parsed: {file_path.name} | "
            f"chapters={len(doc.chapters)} sections={len(doc.sections)} "
            f"words={doc.word_count} images={doc.image_count} tables={doc.table_count}"
        )
        return doc

    def parse_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        glob_pattern: Optional[str] = None,
    ) -> Iterator[ParsedDocument]:
        """
        Parse all documents in a directory.

        Args:
            directory: Directory path
            recursive: Recurse into subdirectories
            glob_pattern: Optional glob pattern (e.g., "**/*.pdf")

        Yields:
            ParsedDocument for each file
        """
        directory = Path(directory)
        if glob_pattern:
            files = list(directory.glob(glob_pattern))
        elif recursive:
            files = [
                f for ext in self.SUPPORTED_FORMATS
                for f in directory.rglob(f"*{ext}")
            ]
        else:
            files = [
                f for ext in self.SUPPORTED_FORMATS
                for f in directory.glob(f"*{ext}")
            ]

        files = sorted(set(files))
        logger.info(f"Found {len(files)} documents in {directory}")

        for i, file_path in enumerate(files):
            logger.info(f"[{i+1}/{len(files)}] Processing: {file_path.name}")
            try:
                yield self.parse(file_path)
            except Exception as e:
                logger.error(f"Failed to parse {file_path.name}: {e}", exc_info=True)

    def parse_batch(
        self,
        file_paths: list[Union[str, Path]],
    ) -> list[ParsedDocument]:
        """Parse a list of documents, returning all results."""
        return list(self.parse_directory.__wrapped__ if False else
                    (self.parse(f) for f in file_paths))

    def to_hf_dataset(
        self,
        documents: Union[list[ParsedDocument], Iterator[ParsedDocument]],
        dataset_name: Optional[str] = None,
        schema: str = "full",
        include_images: bool = True,
        push_to_hub: bool = False,
        hub_repo: Optional[str] = None,
        hub_token: Optional[str] = None,
    ):
        """
        Convert parsed documents to a HuggingFace Dataset.

        Args:
            documents: List or iterator of ParsedDocument objects
            dataset_name: Optional name for the dataset
            schema: "full" | "chunks" | "qa" | "minimal"
                - full: One row per document with all fields
                - chunks: One row per paragraph/section chunk
                - qa: Document + section pairs for QA tasks
                - minimal: Lightweight id/title/text only

        Returns:
            datasets.DatasetDict
        """
        return self._dataset_builder.build(
            documents=documents,
            schema=schema,
            include_images=include_images,
            dataset_name=dataset_name,
            push_to_hub=push_to_hub,
            hub_repo=hub_repo,
            hub_token=hub_token,
        )

    def _compute_doc_id(self, file_path: Path) -> str:
        """Compute stable document ID from file content hash."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    def _count_words(self, paragraphs: list[dict]) -> int:
        """Count total words across paragraphs."""
        return sum(len(p.get("text", "").split()) for p in paragraphs)
