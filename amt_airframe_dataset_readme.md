---
annotations_creators:
- no-annotation
language_creators:
- found
language:
- en
license: other
multilinguality:
- monolingual
size_categories:
- n<1K
task_categories:
- question-answering
- text-retrieval
- image-classification
task_ids:
- extractive-qa
- open-domain-qa
- document-retrieval
---

# AMT Airframe Handbook Dataset

A comprehensive dataset extracted from the FAA Aviation Maintenance Technician (AMT) Airframe Handbook, containing text content and rendered page images suitable for training vision-language models.

[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/Remixonwin/amt-airframe-handbook-dataset)
[![License](https://img.shields.io/badge/License-P proprietary-blue)](LICENSE)

## Overview

This dataset was created using the [doc-parser-engine](https://github.com/Remixonwin/doc-parser-engine) - a production-grade document parsing engine with HuggingFace integration. The source document is the FAA Aviation Maintenance Technician Airframe Handbook, a foundational text for aircraft maintenance professionals.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Pages | 252 |
| Word Count | 80,740 |
| Images | 253 (rendered pages) |
| Paragraphs | 5,521 |
| Sections | 7 |
| Chapters | 1 |

## Data Schema

The dataset contains 19 columns with the following structure:

### Document Metadata
- `doc_id` (string): Unique document identifier
- `title` (string): Document title
- `authors` (list[string]): Document authors
- `doc_type` (string): Document type (e.g., "pdf")
- `source_path` (string): Original file path
- `created_at` (string): Creation timestamp
- `parsed_at` (string): Parsing timestamp

### Content Statistics
- `word_count` (int32): Total word count
- `page_count` (int32): Total page count
- `image_count` (int32): Number of images extracted
- `table_count` (int32): Number of tables extracted

### Structural Content
- `chapters` (list[string]): Chapter titles
- `sections` (struct): Section details including:
  - `section_id`: Section identifier
  - `title`: Section title
  - `level`: Section hierarchy level
  - `content`: Section content
  - `semantic_label`: Semantic classification
  - `page_start`: Starting page number
  - `word_count`: Section word count

### Paragraphs
- `paragraphs` (struct): Paragraph details including:
  - `para_id`: Paragraph identifier (e.g., "para_00001")
  - `text`: Paragraph content
  - `section_id`: Parent section
  - `page`: Page number
  - `word_count`: Paragraph word count

### Tables
- `tables` (struct): Table details including:
  - `table_id`: Table identifier
  - `headers`: Table headers
  - `num_rows`: Number of rows
  - `num_cols`: Number of columns
  - `caption`: Table caption

### Images
- `images` (struct): Image details including:
  - `image_id`: Unique image identifier
  - `caption`: Image caption (when available)
  - `alt_text`: Alternative text
  - `ocr_text`: OCR extracted text
  - `width`: Image width in pixels
  - `height`: Image height in pixels
  - `format`: Image format (PNG)
  - `page`: Source page number
  - `file_path`: File path reference
  - `category`: Image category

### Additional Metadata
- `metadata` (string): Additional metadata
- `footnotes_count` (int32): Number of footnotes
- `references_count` (int32): Number of references

## Usage Example

### Loading the Dataset

```python
from datasets import load_dataset

# Load the dataset from HuggingFace
dataset = load_dataset("Remixonwin/amt-airframe-handbook-dataset")

# Access the training split
train_data = dataset["train"]

# Get the first (and only) document
doc = train_data[0]

print(f"Title: {doc['title']}")
print(f"Word Count: {doc['word_count']}")
print(f"Page Count: {doc['page_count']}")
print(f"Image Count: {doc['image_count']}")
```

### Accessing Paragraphs

```python
# Access structured paragraphs
paragraphs = doc["paragraphs"]
print(f"Total paragraphs: {len(paragraphs['para_id'])}")

# Get first paragraph
first_para = {
    "id": paragraphs["para_id"][0],
    "text": paragraphs["text"][0],
    "page": paragraphs["page"][0],
    "word_count": paragraphs["word_count"][0]
}
print(first_para)
```

### Accessing Images

```python
# Access images with metadata
images = doc["images"]
print(f"Total images: {len(images['image_id'])}")

# Get first image details
first_image = {
    "id": images["image_id"][0],
    "width": images["width"][0],
    "height": images["height"][0],
    "page": images["page"][0],
    "format": images["format"][0]
}
print(first_image)
```

## Image Extraction Details

This dataset uses **page rendering** to capture vector graphics and diagrams from the PDF. The original PDF contains technical drawings, diagrams, and figures rendered as vector graphics which are not extracted by traditional image extraction methods.

- **Rendering DPI**: 150
- **Image Format**: PNG
- **Page Images**: Full-page renders at 1242x1755 pixels

## Use Cases

This dataset is ideal for:

1. **Vision-Language Model Training**: Train multi-modal models on aviation maintenance content
2. **Document Understanding**: Research on technical document parsing and understanding
3. **Question Answering**: Build Q&A systems for aviation maintenance training
4. **Information Retrieval**: Create search systems for technical documentation
5. **Content Analysis**: Analyze structure and content of technical handbooks

## Dataset Creation

The dataset was created using the doc-parser-engine with the following configuration:

```python
from doc_parser_engine.core import DocParserEngine

engine = DocParserEngine(
    enable_captioning=True,
    caption_model="api",
    llm_api_base="http://0.0.0.0:7543",
    llm_model="gpt-4o",
    render_pages=True,    # Enable page rendering
    render_dpi=150       # 150 DPI for rendered images
)

doc = engine.parse("amt_airframe.pdf")

# Create HuggingFace dataset
dataset = engine.to_hf_dataset(
    [doc],
    schema="full",
    include_images=True,
    push_to_hub=True,
    hub_repo="Remixonwin/amt-airframe-handbook-dataset"
)
```

## License

This dataset contains content from the FAA Aviation Maintenance Technician Handbooks. Please verify the appropriate usage rights for your intended application.

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{amt_airframe_handbook_2026,
  author = {Federal Aviation Administration},
  title = {Aviation Maintenance Technician Airframe Handbook},
  year = {2026},
  publisher = {HuggingFace},
  dataset_url = {https://huggingface.co/datasets/Remixonwin/amt-airframe-handbook-dataset}
}
```

## Related Datasets

- [AMT Powerplant Handbook Dataset](https://huggingface.co/datasets/Remixonwin/amt-powerplant-handbook-dataset) - Complementary dataset covering aircraft powerplant systems

## Contact

For questions or issues with the dataset, please open an issue on the [GitHub repository](https://github.com/Remixonwin/doc-parser-engine).

---

*Dataset last updated: February 2026*
