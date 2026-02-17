# DocParserEngine 🔍📄

A production-grade document parsing engine that extracts text, images, and structure, exporting directly to HuggingFace Datasets.

## 🚀 Quick Start

### 1. Installation

```bash
# Install with uv (recommended)
uv sync
```

### 2. Usage

#### Python API
```python
from doc_parser_engine import DocParserEngine

# Initialize
engine = DocParserEngine(enable_ocr=True)

# Parse a document
doc = engine.parse("document.pdf")

# Export to HuggingFace Dataset
dataset = engine.to_hf_dataset([doc], schema="chunks")
```

#### CLI (Recommended)
The engine comes with a powerful CLI for batch processing and dataset creation.

```bash
# Initialize interactive wizard
uv run doc-parser

# Parse with auto-push and intelligent title scouting
uv run doc-parser parse path/to/doc.pdf --push

# Check system status (PyTorch, Tesseract, CUDA)
uv run doc-parser status

# Save persistent default settings
uv run doc-parser config
```

## ✨ Key Features

- **Multi-format Support**: PDF, DOCX, HTML, TXT, MD, EPUB.
- **Enhanced PDF Extraction**: Full image extraction (not just blocks) and structural detection.
- **Intelligent Title Scouting**: Automatically extracts document titles from metadata or headers for descriptive repository names.
- **Automated Image Categorization**: Heuristic-based classification (photo, chart, diagram) using OCR and aspect ratios.
- **AI-Powered**: Optional image captioning (BLIP-2/BLIP) and OCR (Tesseract).
- **Dataset Ready**: Multiple export schemas (full, chunks, minimal, vision).
- **Interactive CLI**: Guided parsing and export workflow with persistent configuration.

## 🏗️ Project Structure

- `src/doc_parser_engine/`:
  - `core.py`: Main engine orchestration.
  - `cli.py`: Typer-based command line interface.
  - `parsers/`: Specialized parsers for each format (PDF, Word, HTML, etc.).
  - `detection/`: Structure and heading detection logic.
  - `dataset/`: HuggingFace Dataset builders and schemas.
  - `extractors/`: Image and table extraction logic.
- `tests/`: Comprehensive test suite verifying all parsers and pipelines.

## 🛠️ Development

### Running Tests
```bash
uv run pytest
```

## 📜 License

Apache 2.0
