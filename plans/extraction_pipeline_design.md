# End-to-End Extraction + Artifact Pipeline Design

**Design Document Version:** 1.0  
**Created:** 2026-02-18  
**Status:** Final  

---

## 1. Overview

This document specifies the complete design for an end-to-end extraction and artifact generation pipeline. The pipeline walks an input folder recursively, extracts content from documents (primarily PDFs with support for other formats), and emits HuggingFace-compatible dataset artifacts under `artifacts/<ISO_TIMESTAMP>/` atomically.

**This design supersedes all conflicting general instructions.** Where this document differs from any prior specifications, this document takes precedence.

---

## 2. Input Specification

### 2.1 Input Folder

- **Default Path:** `/home/remixonwin/Documents/playground/data-parser/prepware_study_guide`
- **Behavior:** Recursively scan all subdirectories
- **Supported File Types:**
  - `.pdf` - Primary format (per-page extraction)
  - `.docx`, `.doc` - One example per document
  - `.html`, `.htm` - One example per document
  - `.txt`, `.md`, `.markdown` - One example per document
  - `.epub` - One example per document

### 2.2 Non-PDF Handling Strategy

| File Type | Extraction Strategy |
|-----------|---------------------|
| PDF | Per-page: one example per page with `sha256(file_path):page_number` ID |
| DOCX, DOC | One example per logical document |
| HTML, HTXT | One example per file |
| TXT, MD, MARKDOWN | One example per file |
| EPUB | One example per file (chapter-level optional) |

---

## 3. Directory Layout

```
artifacts/
└── {ISO_TIMESTAMP}/
    ├── metadata.json
    ├── manifest.json
    ├── dataset_info.json
    ├── verify_outputs.txt
    ├── audit_report.json
    ├── data/
    │   ├── train/
    │   │   ├── data-00000-of-00001.parquet
    │   │   ├── data-00000-of-00002.parquet
    │   │   └── ...
    │   ├── validation/
    │   │   └── ...
    │   └── test/
    │       └── ...
    ├── jsonl/
    │   ├── train/
    │   │   ├── shard-00000.jsonl
    │   │   ├── shard-00001.jsonl
    │   │   └── ...
    │   ├── validation/
    │   │   └── ...
    │   └── test/
    │       └── ...
    ├── images/
    │   ├── {doc_id}/
    │   │   ├── page_001.png
    │   │   ├── page_002.png
    │   │   └── ...
    │   └── ...
    └── logs/
        ├── extraction.log
        └── warnings.log
```

### 3.1 Naming Conventions

| Element | Format | Example |
|---------|--------|---------|
| Root timestamp directory | `YYYYMMDDTHHMMSSZ` | `20260218T013000Z` |
| Parquet shards | `data-XXXXX-of-YYYYY.parquet` | `data-00000-of-00010.parquet` |
| JSONL shards | `shard-XXXXX.jsonl` | `shard-00000.jsonl` |
| Image files | `{page_number:03d}.{ext}` | `001.png`, `002.jpg` |
| Log files | `{operation}.log` | `extraction.log` |

---

## 4. Deterministic ID Format

### 4.1 Primary ID Scheme

```
example_id = sha256(normalized_file_path) + ":" + str(page_number)
```

**Components:**
- `normalized_file_path`: Absolute path normalized (resolved symlinks, normalized separators, lowercase on Windows)
- `page_number`: 1-based integer for PDFs; `1` for non-PDF documents

**Example:**
```
# For PDF at /home/user/docs/report.pdf, page 5
example_id = "a1b2c3d4e5f6...:5"
```

### 4.2 ID Generation Algorithm

```python
import hashlib
import os

def generate_example_id(file_path: str, page_number: int = 1) -> str:
    """
    Generate deterministic example ID.
    
    Args:
        file_path: Absolute normalized file path
        page_number: 1-based page number (1 for non-PDFs)
    
    Returns:
        Deterministic ID string
    """
    # Normalize path: resolve symlinks, use forward slashes
    normalized = os.path.normpath(os.path.abspath(file_path))
    
    # Compute SHA-256 hash (full hash for uniqueness)
    h = hashlib.sha256(normalized.encode('utf-8'))
    hash_prefix = h.hexdigest()
    
    return f"{hash_prefix}:{page_number}"
```

---

## 5. JSONL Schema

### 5.1 Per-Page Record Schema

```json
{
  "example_id": "sha256hash:pageno",
  "doc_id": "sha256hash",
  "file_path": "/absolute/path/to/document.pdf",
  "file_name": "document.pdf",
  "file_extension": ".pdf",
  "page_number": 1,
  "total_pages": 42,
  "doc_type": "pdf",
  "title": "Document Title",
  "authors": ["Author One", "Author Two"],
  "created_at": "2024-01-15T10:30:00Z",
  "parsed_at": "2026-02-18T01:30:00Z",
  "extracted_text": "Full text content of the page...",
  "extracted_text_length": 1234,
  "word_count": 187,
  "images": [
    {
      "image_id": "img_001",
      "page": 1,
      "bbox": [100, 200, 400, 500],
      "width": 300,
      "height": 300,
      "format": "png",
      "caption": "Image description",
      "alt_text": "Alt text"
    }
  ],
  "tables": [
    {
      "table_id": "tbl_001",
      "page": 1,
      "bbox": [50, 100, 550, 300],
      "headers": ["Col1", "Col2", "Col3"],
      "rows": [["a", "b", "c"], ["d", "e", "f"]],
      "caption": "Table caption"
    }
  ],
  "metadata": {
    "subject": "Subject",
    "creator": "Creator",
    "producer": "Producer",
    "ocr_applied": false,
    "extraction_warnings": []
  },
  "extraction_warnings": []
}
```

### 5.2 JSONL File Structure

- **Shard Size:** Maximum 10,000 records per shard
- **Naming:** `shard-00000.jsonl`, `shard-00001.jsonl`, etc.
- **Compression:** None (plain JSONL for debugging); gzip optional for production
- **Line Ending:** Unix newline (`\n`)
- **Encoding:** UTF-8 with BOM optional

---

## 6. Parquet Schema

### 6.1 HuggingFace Features Definition

```python
from datasets import Features, Value, Sequence, Image

PER_PAGE_FEATURES = Features({
    # Identification
    "example_id": Value("string"),
    "doc_id": Value("string"),
    "file_path": Value("string"),
    "file_name": Value("string"),
    "file_extension": Value("string"),
    "page_number": Value("int32"),
    "total_pages": Value("int32"),
    "doc_type": Value("string"),
    
    # Document Metadata
    "title": Value("string"),
    "authors": Sequence(Value("string")),
    "created_at": Value("string"),
    "parsed_at": Value("string"),
    
    # Text Content
    "extracted_text": Value("string"),
    "extracted_text_length": Value("int32"),
    "word_count": Value("int32"),
    
    # Images
    "images": Sequence({
        "image_id": Value("string"),
        "page": Value("int32"),
        "bbox": Sequence(Value("float32")),
        "width": Value("int32"),
        "height": Value("int32"),
        "format": Value("string"),
        "caption": Value("string"),
        "alt_text": Value("string"),
    }),
    
    # Tables
    "tables": Sequence({
        "table_id": Value("string"),
        "page": Value("int32"),
        "bbox": Sequence(Value("float32")),
        "headers": Sequence(Value("string")),
        "rows": Sequence(Sequence(Value("string"))),
        "caption": Value("string"),
    }),
    
    # Metadata
    "metadata": Value("string"),  # JSON string
    "extraction_warnings": Sequence(Value("string")),
})
```

### 6.2 Sharding Strategy

| Total Records | Num Shards | Records per Shard |
|---------------|------------|-------------------|
| 1 - 10,000 | 1 | ≤ 10,000 |
| 10,001 - 100,000 | 10 | ≤ 10,000 |
| 100,001 - 1,000,000 | 100 | ≤ 10,000 |

---

## 7. Dataset Info Content

### 7.1 dataset_info.json Structure

```json
{
  "citation": "",
  "description": "Extracted from prepware_study_guide documents. Each row represents one page from a source document.",
  "features": {
    "example_id": {"dtype": "string", "_type": "Value"},
    "doc_id": {"dtype": "string", "_type": "Value"},
    "file_path": {"dtype": "string", "_type": "Value"},
    "file_name": {"dtype": "string", "_type": "Value"},
    "file_extension": {"dtype": "string", "_type": "Value"},
    "page_number": {"dtype": "int32", "_type": "Value"},
    "total_pages": {"dtype": "int32", "_type": "Value"},
    "doc_type": {"dtype": "string", "_type": "Value"},
    "title": {"dtype": "string", "_type": "Value"},
    "authors": {"feature": {"dtype": "string", "_type": "Value"}, "_type": "List"},
    "created_at": {"dtype": "string", "_type": "Value"},
    "parsed_at": {"dtype": "string", "_type": "Value"},
    "extracted_text": {"dtype": "string", "_type": "Value"},
    "extracted_text_length": {"dtype": "int32", "_type": "Value"},
    "word_count": {"dtype": "int32", "_type": "Value"},
    "images": {
        "feature": {
            "image_id": {"dtype": "string", "_type": "Value"},
            "page": {"dtype": "int32", "_type": "Value"},
            "bbox": {"feature": {"dtype": "float32", "_type": "Value"}, "_type": "List"},
            "width": {"dtype": "int32", "_type": "Value"},
            "height": {"dtype": "int32", "_type": "Value"},
            "format": {"dtype": "string", "_type": "Value"},
            "caption": {"dtype": "string", "_type": "Value"},
            "alt_text": {"dtype": "string", "_type": "Value"},
        },
        "_type": "List"
    },
    "tables": {...},
    "metadata": {"dtype": "string", "_type": "Value"},
    "extraction_warnings": {"feature": {"dtype": "string", "_type": "Value"}, "_type": "List"}
  },
  "homepage": "",
  "license": "",
  "annotations_creators": ["no-annotations"],
  "language": ["en"],
  "language_details": "English",
  "multilingual": false,
  "size_categories": ["n<1K", "1K<n<10K", "10K<n<100K"],
  "source_data": ["original"],
  "task_categories": ["feature-extraction"],
  "task_ids": ["extractive-qa", "table-to-text"]
}
```

---

## 8. Per-Page PDF Extraction

### 8.1 Extraction Process

1. **Open PDF** using PyMuPDF (fitz)
2. **Extract Metadata** from document properties
3. **Iterate Pages** (1-based indexing)
4. **Per-Page Extraction:**
   - Extract text blocks with position data
   - Extract images with bounding boxes
   - Detect and extract tables
   - Apply OCR if page is scanned (< 10 words)
5. **Generate Records** one per page

### 8.2 Page-Level Data Fields

```python
@dataclass
class PageRecord:
    """Single page extraction result."""
    example_id: str          # sha256(path):page
    doc_id: str              # sha256(path)[:16]
    file_path: str          # Absolute path
    file_name: str          # Basename
    file_extension: str     # .pdf
    page_number: int         # 1-based
    total_pages: int         # Document total
    doc_type: str           # "pdf"
    title: str              # Document title
    authors: list[str]      # Author list
    created_at: str         # Creation date
    parsed_at: str          # Extraction timestamp
    extracted_text: str     # Full page text
    extracted_text_length: int
    word_count: int
    images: list[ImageRecord]
    tables: list[TableRecord]
    metadata: dict
    extraction_warnings: list[str]
```

### 8.3 Image Extraction Details

- **Method:** PyMuPDF `page.get_images(full=True)` + `doc.extract_image(xref)`
- **Deduplication:** By xref to avoid duplicate extractions
- **Minimum Size:** Configurable (default: 100px width/height)
- **Formats:** Preserve original format (PNG, JPEG, etc.)
- **Storage:** `images/{doc_id}/{page:03d}.{format}`

### 8.4 Table Extraction

- **Method:** Structure detector + heuristic analysis
- **Output:** Structured table with headers and rows
- **Bounding Box:** From layout analysis

---

## 9. Extraction Warnings Strategy

### 9.1 Warning Categories

| Code | Category | Description |
|------|----------|-------------|
| `W001` | OCR_APPLIED | OCR was applied due to low text content |
| `W002` | LOW_QUALITY_TEXT | Text quality below threshold |
| `W003` | ENCRYPTED_PDF | PDF is password encrypted |
| `W004` | CORRUPTED_PAGE | Page failed to render properly |
| `W005` | MISSING_FONTS | Fonts not embedded, layout may be incorrect |
| `W006` | LARGE_IMAGE | Image exceeds size threshold |
| `W007` | EXTRACTION_TIMEOUT | Page extraction exceeded timeout |
| `W008` | UNSUPPORTED_COMPRESSION | PDF uses unsupported compression |

### 9.2 Warning Storage

Warnings are stored per-example in the `extraction_warnings` array:

```json
{
  "extraction_warnings": ["W001:OCR_APPLIED", "W006:LARGE_IMAGE:2048x2048"]
}
```

---

## 10. Stable Ordering Rules

### 10.1 File Ordering

Files MUST be processed in **deterministic alphabetical order**:

1. **Recursive traversal** of input directory
2. **Sort by normalized path** (forward slashes, case-sensitive)
3. **Stable sort** - if same path, preserve original order

### 10.2 Page Ordering

Within each document, pages are ordered by page number (1, 2, 3, ...).

### 10.3 Shard Ordering

Records are written to shards sequentially:
- Shard 0 receives records 0-9999
- Shard 1 receives records 10000-19999
- etc.

---

## 11. Concurrency Model

### 11.1 Thread/Process Design

```
                    ┌─────────────────┐
                    │   Main Process  │
                    │ (Orchestrator)  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼────┐  ┌──────▼──────┐  ┌────▼────────┐
     │  Worker 1   │  │  Worker 2   │  │  Worker N   │
     │ (File 1-N)  │  │ (File N+1)  │  │ (File ...)  │
     └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
            │                │                │
            └────────────────┼────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Result Queue   │
                    │ (Thread-Safe)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Writer Thread  │
                    │ (JSONL/Parquet) │
                    └─────────────────┘
```

### 11.2 Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_workers` | `min(8, cpu_count())` | Maximum parallel workers |
| `queue_size` | 1000 | Max pending items in queue |
| `per_file_concurrency` | 1 | Pages processed sequentially per file |

### 11.3 Thread Safety

- **File-level parallelism:** Different files can be processed in parallel
- **Page-level:** Pages within same file processed sequentially (to maintain order)
- **Queue:** Thread-safe queue for results
- **Writing:** Single writer thread for deterministic output

---

## 12. Retry/Timeout Configuration

### 12.1 File I/O Timeouts

| Operation | Timeout | Retry Count |
|-----------|---------|-------------|
| File open/read | 30s | 3 |
| File write | 30s | 3 |
| Directory create | 10s | 1 |
| Atomic rename | 10s | 3 |

### 12.2 Extraction Timeouts

| Operation | Timeout | Retry Count |
|-----------|---------|-------------|
| PDF page parse | 60s | 2 |
| Image extraction | 30s | 2 |
| OCR processing | 120s | 1 |
| Table detection | 30s | 2 |
| Full document | 600s (10min) | 1 |

### 12.3 Retry Strategy

```python
import time
from functools import wraps

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential: bool = True
):
    """Retry decorator with exponential backoff."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = min(
                            base_delay * (2 ** attempt) if exponential else base_delay,
                            max_delay
                        )
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator
```

---

## 13. Atomic Write Procedure

### 13.1 Temporary Directory Strategy

```
artifacts/
└── {ISO_TIMESTAMP}.tmp/          # Temporary directory
    ├── data/
    │   └── ...
    ├── jsonl/
    │   └── ...
    └── ...
         │
         │ (all writes complete)
         │
         ▼
artifacts/
└── {ISO_TIMESTAMP}/              # Atomic rename
    └── ...
```

### 13.2 Write Protocol

1. **Create tmp directory:** `artifacts/{ISO_TIMESTAMP}.tmp`
2. **Write all data** to tmp directory
3. **Verify checksums** of all written files
4. **Generate manifest** with all file checksums
5. **Atomic rename:** `mv {ISO_TIMESTAMP}.tmp {ISO_TIMESTAMP}`
6. **Create metadata.json** with run information

### 13.3 Implementation

```python
import os
import shutil
import tempfile
import hashlib
from pathlib import Path
from datetime import datetime

def atomic_write_artifacts(
    output_base: Path,
    timestamp: str,
    data_generator
) -> Path:
    """
    Atomically write artifacts using tmp-dir pattern.
    
    Args:
        output_base: Base artifacts directory
        timestamp: ISO timestamp string
        data_generator: Iterator yielding (relative_path, content) tuples
    
    Returns:
        Final artifacts directory path
    """
    tmp_dir = output_base / f"{timestamp}.tmp"
    final_dir = output_base / timestamp
    
    # Step 1: Create tmp directory
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 2: Write all data
        for rel_path, content in data_generator:
            file_path = tmp_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temp file first
            fd, temp_path = tempfile.mkstemp(
                dir=tmp_dir,
                prefix='.tmp_'
            )
            try:
                os.write(fd, content)
                os.close(fd)
                # Atomic rename
                os.replace(temp_path, str(file_path))
            except:
                os.close(fd)
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise
        
        # Step 3: Generate manifest
        manifest = generate_manifest(tmp_dir)
        manifest_path = tmp_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Step 4: Atomic rename to final location
        os.replace(str(tmp_dir), str(final_dir))
        
    except Exception:
        # Cleanup on failure
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    
    return final_dir
```

---

## 14. Manifest with Checksums

### 14.1 Manifest Structure

```json
{
  "timestamp": "20260218T013000Z",
  "generated_at": "2026-02-18T01:30:00.000Z",
  "input_directory": "/home/user/docs",
  "total_files": 29,
  "total_pages": 1247,
  "total_examples": 1247,
  "shard_count": 2,
  "artifacts": [
    {
      "path": "data/train/data-00000-of-00002.parquet",
      "size": 1234567,
      "sha256": "a1b2c3d4e5f6...",
      "num_records": 10000
    },
    {
      "path": "data/train/data-00001-of-00002.parquet",
      "size": 987654,
      "sha256": "f6e5d4c3b2a1...",
      "num_records": 247
    },
    {
      "path": "jsonl/train/shard-00000.jsonl",
      "size": 2345678,
      "sha256": "...",
      "num_records": 10000
    }
  ]
}
```

### 14.2 Checksum Algorithm

```python
import hashlib

def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()
```

---

## 15. Verification Commands

### 15.1 Verification Script Execution

After artifact generation, run verification commands and capture output to `verify_outputs.txt`.

### 15.2 Verification Commands

```bash
# 1. Validate JSONL structure
cd artifacts/20260218T013000Z
python3 -c "
import json
import sys
errors = []
for i, line in enumerate(open('jsonl/train/shard-00000.jsonl')):
    try:
        rec = json.loads(line)
        if 'example_id' not in rec:
            errors.append(f'Line {i}: missing example_id')
    except json.JSONDecodeError as e:
        errors.append(f'Line {i}: {e}')
if errors:
    for e in errors[:10]:
        print(e)
    sys.exit(1)
print('JSONL validation: PASSED')
"

# 2. Validate Parquet can be read
python3 -c "
import pandas as pd
df = pd.read_parquet('data/train/data-00000-of-00002.parquet')
print(f'Parquet validation: PASSED ({len(df)} rows)')
"

# 3. Verify deterministic IDs
python3 -c "
import json
import hashlib
ids = set()
for line in open('jsonl/train/shard-00000.jsonl'):
    rec = json.loads(line)
    eid = rec['example_id']
    if eid in ids:
        print(f'Duplicate ID: {eid}')
        exit(1)
    ids.add(eid)
print(f'ID uniqueness: PASSED ({len(ids)} unique IDs)')
"

# 4. Verify file counts match manifest
python3 -c "
import json
with open('manifest.json') as f:
    m = json.load(f)
print(f'Manifest validation: PASSED ({m[\"total_examples\"]} examples)')
"

# 5. Verify images exist
python3 -c "
from pathlib import Path
img_dir = Path('images')
if img_dir.exists():
    img_count = len(list(img_dir.rglob('*.*')))
    print(f'Image validation: PASSED ({img_count} images)')
else:
    print('Image validation: PASSED (0 images)')
"
```

### 15.3 verify_outputs.txt Format

```
=== Verification Output ===
Timestamp: 20260218T013000Z
Started: 2026-02-18T01:30:00.000Z
Completed: 20260218T01:30:15.123Z

--- Command 1: JSONL Validation ---
Command: python3 -c "..."
Output:
JSONL validation: PASSED
Exit Code: 0

--- Command 2: Parquet Validation ---
Command: python3 -c "..."
Output:
Parquet validation: PASSED (10000 rows)
Exit Code: 0

--- Command 3: ID Uniqueness ---
Command: python3 -c "..."
Output:
ID uniqueness: PASSED (10000 unique IDs)
Exit Code: 0

--- Command 4: Manifest Validation ---
Command: python3 -c "..."
Output:
Manifest validation: PASSED (1247 examples)
Exit Code: 0

--- Command 5: Image Validation ---
Command: python3 -c "..."
Output:
Image validation: PASSED (523 images)
Exit Code: 0

=== Summary ===
All verifications PASSED
Total Time: 15.123s
```

---

## 16. Audit Report Schema

### 16.1 audit_report.json Structure

Matches existing schema from `artifacts/20260218T011100Z/audit_report.json`:

```json
{
  "run_id": "uuid-v4-string",
  "repo_id": "username/dataset-name",
  "timestamp": "2026-02-18T01:30:00.000000Z",
  "cli_args": {
    "sample_size": 5,
    "seed": 42,
    "similarity_threshold": 0.6
  },
  "environment": {
    "python_version": "3.12.3",
    "platform": "Linux-6.17...",
    "hf_token_present": true
  },
  "dataset_summary": {
    "num_rows": 1247,
    "num_files": 29,
    "formats": ["pdf"],
    "sample_records_count": 5
  },
  "quality_checks": [
    {
      "name": "pass_rate_vs_threshold",
      "status": "pass|fail",
      "message": "Pass rate 0.85 >= threshold 0.6",
      "metrics": {
        "pass_rate": 0.85,
        "threshold": 0.6
      }
    }
  ],
  "artifacts": [
    {
      "path": "data/train/data-00000-of-00002.parquet",
      "size": 1234567,
      "sha256": "a1b2c3d4..."
    }
  ],
  "warnings": [],
  "errors": [],
  "exit_code": 0
}
```

---

## 17. Implementation Priority

### Phase 1: Core Extraction
1. Directory walking with stable ordering
2. PDF per-page extraction
3. JSONL shard writing
4. Deterministic ID generation
5. Basic metadata extraction

### Phase 2: Output Generation
6. Parquet conversion
7. Image extraction and storage
8. Table extraction
9. Manifest generation with checksums

### Phase 3: Quality Assurance
10. Verification commands execution
11. verify_outputs.txt generation
12. audit_report.json generation
13. Atomic write procedure

### Phase 4: Production Features
14. Concurrency model
15. Retry/timeouts
16. Extraction warnings
17. Non-PDF support

---

## 18. File Summary

| File | Description |
|------|-------------|
| `artifacts/{TS}/metadata.json` | Run metadata (timestamp, version, input path) |
| `artifacts/{TS}/manifest.json` | All output files with checksums |
| `artifacts/{TS}/dataset_info.json` | HF dataset metadata |
| `artifacts/{TS}/verify_outputs.txt` | Verification command stdout/stderr |
| `artifacts/{TS}/audit_report.json` | Audit results matching schema |
| `artifacts/{TS}/data/*/*.parquet` | HuggingFace parquet shards |
| `artifacts/{TS}/jsonl/*/*.jsonl` | Raw JSONL shards |
| `artifacts/{TS}/images/*/*` | Extracted images |

---

*End of Design Document*
