"""
DatasetBuilder - Converts ParsedDocument objects into HuggingFace Datasets.

Supported schemas:
- "full"    : One row per document, all fields
- "chunks"  : One row per paragraph/section chunk (for RAG/QA)
- "images"  : One row per image with captions
- "minimal" : Lightweight text-only dataset
- "qa"      : Question-answering dataset (context, question, answer)

## Implementation Plan
- Highlight enhanced PDF extraction (paragraph merging, heading heuristics).
- Update the project structure to accurately reflect the `src` layout.
- Add a "Development" section with testing instructions (pytest).

## Hub Export Automation
- **Credential Handling**: Ensure `HF_TOKEN` is used from `.env` without prompts.
- **Intelligent Naming**: Generate `repo_id` based on `HF_USERNAME` (.env) and the input filename/directory.
- **CLI Refinement**: Remove the interactive prompt for Repository ID if `HF_USERNAME` is available.
"""

import logging
from typing import Optional, Union, Iterator

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Builds HuggingFace-compatible datasets from ParsedDocument objects.

    Integrates directly with the datasets library and supports:
    - Multiple schema formats
    - Image inclusion as PIL/bytes
    - Automatic feature type inference
    - Push to HuggingFace Hub
    - DatasetDict with train/val/test splits
    """

    SCHEMAS = ("full", "chunks", "images", "minimal", "qa")

    def build(
        self,
        documents,
        schema: str = "full",
        include_images: bool = True,
        dataset_name: Optional[str] = None,
        push_to_hub: bool = False,
        hub_repo: Optional[str] = None,
        hub_token: Optional[str] = None,
    ):
        """
        Build a HuggingFace Dataset from parsed documents.

        Args:
            documents: List or iterator of ParsedDocument
            schema: Schema format ("full", "chunks", "images", "minimal", "qa")
            include_images: Include image bytes in dataset
            dataset_name: Optional dataset name
            push_to_hub: Push to HuggingFace Hub
            hub_repo: Hub repo ID
            hub_token: HF API token

        Returns:
            datasets.DatasetDict
        """
        try:
            import datasets
        except ImportError:
            raise RuntimeError("datasets not installed. pip install datasets")

        if schema not in self.SCHEMAS:
            raise ValueError(f"Schema must be one of {self.SCHEMAS}")

        # Collect documents
        docs_list = list(documents) if not isinstance(documents, list) else documents
        logger.info(f"Building {schema} dataset from {len(docs_list)} documents")

        # Build records based on schema
        if schema == "full":
            records = self._build_full(docs_list, include_images)
            features = self._get_full_features(include_images)
        elif schema == "chunks":
            records = self._build_chunks(docs_list)
            features = self._get_chunks_features()
        elif schema == "images":
            records = self._build_images(docs_list, include_images)
            features = self._get_images_features(include_images)
        elif schema == "minimal":
            records = self._build_minimal(docs_list)
            features = self._get_minimal_features()
        elif schema == "qa":
            records = self._build_qa(docs_list)
            features = self._get_qa_features()

        if not records:
            logger.warning("No records generated!")
            return datasets.DatasetDict()

        # Create dataset
        dataset = datasets.Dataset.from_list(records, features=features)

        # Create DatasetDict with splits
        split_result = self._create_splits(dataset)

        if push_to_hub and hub_repo:
            logger.info(f"Pushing to Hub: {hub_repo}")

            # Generate and add a high-quality README.md (Dataset Card)
            readme_content = self._generate_readme(
                hub_repo=hub_repo,
                schema=schema,
                num_docs=len(docs_list),
                num_records=len(records),
            )

            split_result.push_to_hub(
                hub_repo,
                token=hub_token,
                private=False,
                commit_message=f"Dataset: {dataset_name or 'DocParserEngine output'}",
            )

            # Push README as a separate file to ensure it's in the root
            try:
                from huggingface_hub import HfApi

                api = HfApi(token=hub_token)
                api.upload_file(
                    path_or_fileobj=readme_content.encode("utf-8"),
                    path_in_repo="README.md",
                    repo_id=hub_repo,
                    repo_type="dataset",
                )
                logger.debug("README.md (Dataset Card) uploaded successfully")
            except Exception as e:
                logger.warning(f"Failed to upload README.md: {e}")

            logger.info(
                f"Dataset pushed to: https://huggingface.co/datasets/{hub_repo}"
            )

        return split_result

    # -------------------------------------------------------------------------
    # Schema builders
    # -------------------------------------------------------------------------

    def _build_full(self, docs: list, include_images: bool) -> list[dict]:
        """One row per document."""
        records = []
        for doc in docs:
            d = doc.to_dict()
            record = {
                "doc_id": d["doc_id"],
                "title": d["title"],
                "authors": d["authors"],
                "doc_type": d["doc_type"],
                "source_path": d["source_path"],
                "created_at": d["created_at"],
                "parsed_at": d["parsed_at"],
                # Counts
                "word_count": d["word_count"],
                "page_count": d["page_count"],
                "image_count": d["image_count"],
                "table_count": d["table_count"],
                # Text content
                "chapters": [ch.get("title", "") for ch in d["chapters"]],
                "sections": self._columnarize(
                    [
                        {
                            "section_id": s["section_id"],
                            "title": s["title"],
                            "level": s["level"],
                            "content": s["content"][:50000],
                            "semantic_label": s.get("semantic_label") or "",
                            "page_start": s["page_start"],
                            "word_count": s["word_count"],
                        }
                        for s in d["sections"]
                    ]
                ),
                "paragraphs": self._columnarize(
                    [
                        {
                            "para_id": p["para_id"],
                            "text": p["text"],
                            "section_id": p["section_id"],
                            "page": p["page"],
                            "word_count": p["word_count"],
                        }
                        for p in d["paragraphs"]
                    ]
                ),
                # Tables
                "tables": self._columnarize(
                    [
                        {
                            "table_id": t.get("table_id", ""),
                            "headers": t.get("headers", []),
                            "num_rows": t.get("num_rows", 0),
                            "num_cols": t.get("num_cols", 0),
                            "caption": t.get("caption", ""),
                        }
                        for t in d["tables"]
                    ]
                ),
                # Images (metadata only, or with bytes)
                "images": self._columnarize(
                    [self._image_record(img, include_images) for img in d["images"]]
                ),
                # Metadata
                "metadata": str(d["metadata"]),
                "footnotes_count": len(d["footnotes"]),
                "references_count": len(d["references"]),
            }
            records.append(record)
        return records

    def _build_chunks(self, docs: list) -> list[dict]:
        """One row per paragraph chunk - ideal for RAG/embedding."""
        records = []
        for doc in docs:
            d = doc.to_dict()
            # Build section lookup
            section_map = {s["section_id"]: s for s in d["sections"]}

            for para in d["paragraphs"]:
                sec = section_map.get(para["section_id"], {})
                records.append(
                    {
                        "chunk_id": f"{d['doc_id']}_{para['para_id']}",
                        "doc_id": d["doc_id"],
                        "doc_title": d["title"],
                        "doc_type": d["doc_type"],
                        "section_title": sec.get("title", ""),
                        "section_level": sec.get("level", 0),
                        "semantic_label": sec.get("semantic_label") or "",
                        "text": para["text"],
                        "page": para["page"],
                        "word_count": para["word_count"],
                        "char_count": len(para["text"]),
                    }
                )
        return records

    def _build_images(self, docs: list, include_images: bool) -> list[dict]:
        """One row per image - for vision/captioning tasks."""
        records = []
        for doc in docs:
            d = doc.to_dict()
            for img in d["images"]:
                record = {
                    "image_id": img["image_id"],
                    "doc_id": d["doc_id"],
                    "doc_title": d["title"],
                    "caption": img.get("caption", ""),
                    "caption_model": img.get("caption_model", ""),
                    "alt_text": img.get("alt_text", ""),
                    "ocr_text": img.get("ocr_text", ""),
                    "width": img.get("width", 0),
                    "height": img.get("height", 0),
                    "format": img.get("format", ""),
                    "page": img.get("page", 0),
                    "file_path": img.get("file_path", ""),
                }
                if include_images:
                    record["image"] = self._load_pil(img)
                records.append(record)
        return records

    def _build_minimal(self, docs: list) -> list[dict]:
        """Minimal text-only dataset."""
        records = []
        for doc in docs:
            d = doc.to_dict()
            full_text = " ".join(p["text"] for p in d["paragraphs"])
            records.append(
                {
                    "doc_id": d["doc_id"],
                    "title": d["title"],
                    "authors": ", ".join(d["authors"]),
                    "text": full_text,
                    "word_count": d["word_count"],
                    "doc_type": d["doc_type"],
                }
            )
        return records

    def _build_qa(self, docs: list) -> list[dict]:
        """Section-level QA pairs - context/title for question generation."""
        records = []
        for doc in docs:
            d = doc.to_dict()
            for section in d["sections"]:
                if not section["content"] or section["word_count"] < 20:
                    continue
                records.append(
                    {
                        "qa_id": f"{d['doc_id']}_{section['section_id']}",
                        "doc_id": d["doc_id"],
                        "doc_title": d["title"],
                        "section_title": section["title"],
                        "section_level": section["level"],
                        "semantic_label": section.get("semantic_label") or "",
                        "context": section["content"][:10000],
                        "page_start": section["page_start"],
                        "word_count": section["word_count"],
                    }
                )
        return records

    # -------------------------------------------------------------------------
    # Feature schemas
    # -------------------------------------------------------------------------

    def _get_full_features(self, include_images: bool):
        import datasets

        return datasets.Features(
            {
                "doc_id": datasets.Value("string"),
                "title": datasets.Value("string"),
                "authors": datasets.Sequence(datasets.Value("string")),
                "doc_type": datasets.Value("string"),
                "source_path": datasets.Value("string"),
                "created_at": datasets.Value("string"),
                "parsed_at": datasets.Value("string"),
                "word_count": datasets.Value("int32"),
                "page_count": datasets.Value("int32"),
                "image_count": datasets.Value("int32"),
                "table_count": datasets.Value("int32"),
                "chapters": datasets.Sequence(datasets.Value("string")),
                "sections": datasets.Sequence(
                    {
                        "section_id": datasets.Value("string"),
                        "title": datasets.Value("string"),
                        "level": datasets.Value("int32"),
                        "content": datasets.Value("string"),
                        "semantic_label": datasets.Value("string"),
                        "page_start": datasets.Value("int32"),
                        "word_count": datasets.Value("int32"),
                    }
                ),
                "paragraphs": datasets.Sequence(
                    {
                        "para_id": datasets.Value("string"),
                        "text": datasets.Value("string"),
                        "section_id": datasets.Value("string"),
                        "page": datasets.Value("int32"),
                        "word_count": datasets.Value("int32"),
                    }
                ),
                "tables": datasets.Sequence(
                    {
                        "table_id": datasets.Value("string"),
                        "headers": datasets.Sequence(datasets.Value("string")),
                        "num_rows": datasets.Value("int32"),
                        "num_cols": datasets.Value("int32"),
                        "caption": datasets.Value("string"),
                    }
                ),
                "images": datasets.Sequence(
                    {
                        "image_id": datasets.Value("string"),
                        "caption": datasets.Value("string"),
                        "alt_text": datasets.Value("string"),
                        "ocr_text": datasets.Value("string"),
                        "width": datasets.Value("int32"),
                        "height": datasets.Value("int32"),
                        "format": datasets.Value("string"),
                        "page": datasets.Value("int32"),
                        "file_path": datasets.Value("string"),
                        "category": datasets.Value("string"),
                    }
                ),
                "metadata": datasets.Value("string"),
                "footnotes_count": datasets.Value("int32"),
                "references_count": datasets.Value("int32"),
            }
        )

    def _get_chunks_features(self):
        import datasets

        return datasets.Features(
            {
                "chunk_id": datasets.Value("string"),
                "doc_id": datasets.Value("string"),
                "doc_title": datasets.Value("string"),
                "doc_type": datasets.Value("string"),
                "section_title": datasets.Value("string"),
                "section_level": datasets.Value("int32"),
                "semantic_label": datasets.Value("string"),
                "text": datasets.Value("string"),
                "page": datasets.Value("int32"),
                "word_count": datasets.Value("int32"),
                "char_count": datasets.Value("int32"),
            }
        )

    def _get_images_features(self, include_images: bool):
        import datasets

        features = {
            "image_id": datasets.Value("string"),
            "doc_id": datasets.Value("string"),
            "doc_title": datasets.Value("string"),
            "caption": datasets.Value("string"),
            "caption_model": datasets.Value("string"),
            "alt_text": datasets.Value("string"),
            "ocr_text": datasets.Value("string"),
            "width": datasets.Value("int32"),
            "height": datasets.Value("int32"),
            "format": datasets.Value("string"),
            "page": datasets.Value("int32"),
            "file_path": datasets.Value("string"),
        }
        if include_images:
            features["image"] = datasets.Image()
        return datasets.Features(features)

    def _get_minimal_features(self):
        import datasets

        return datasets.Features(
            {
                "doc_id": datasets.Value("string"),
                "title": datasets.Value("string"),
                "authors": datasets.Value("string"),
                "text": datasets.Value("string"),
                "word_count": datasets.Value("int32"),
                "doc_type": datasets.Value("string"),
            }
        )

    def _get_qa_features(self):
        import datasets

        return datasets.Features(
            {
                "qa_id": datasets.Value("string"),
                "doc_id": datasets.Value("string"),
                "doc_title": datasets.Value("string"),
                "section_title": datasets.Value("string"),
                "section_level": datasets.Value("int32"),
                "semantic_label": datasets.Value("string"),
                "context": datasets.Value("string"),
                "page_start": datasets.Value("int32"),
                "word_count": datasets.Value("int32"),
            }
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _columnarize(self, list_of_dicts: list[dict]) -> dict:
        """Convert a list of dicts to a dict of lists (required for nested HF schemas)."""
        if not list_of_dicts:
            return {}
        # Get all keys from all dicts to handle sparse data
        keys = set()
        for d in list_of_dicts:
            keys.update(d.keys())

        return {k: [d.get(k) for d in list_of_dicts] for k in keys}

    def _image_record(self, img: dict, include_image_bytes: bool) -> dict:
        record = {
            "image_id": img.get("image_id", ""),
            "caption": img.get("caption", ""),
            "alt_text": img.get("alt_text", ""),
            "ocr_text": img.get("ocr_text", ""),
            "width": img.get("width", 0),
            "height": img.get("height", 0),
            "format": img.get("format", ""),
            "page": img.get("page", 0),
            "file_path": img.get("file_path", ""),
            "category": img.get("category", "photo"),
        }
        return record

    def _load_pil(self, img_dict: dict):
        """Load image bytes as PIL Image for datasets.Image() feature."""
        try:
            from PIL import Image
            import io

            data = img_dict.get("image_bytes", b"")
            if data:
                return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            pass
        return None

    def _generate_readme(
        self, hub_repo: str, schema: str, num_docs: int, num_records: int
    ) -> str:
        """Generate a high-quality Hugging Face Dataset Card (README.md)."""
        repo_name = hub_repo.split("/")[-1].replace("-", " ").title()

        # Determine splits to include in YAML
        splits_yaml = ""
        if num_records < 4:
            splits_yaml = """  - split: train
    path: "data/train-*" """
        else:
            splits_yaml = """  - split: train
    path: "data/train-*"
  - split: validation
    path: "data/validation-*"
  - split: test
    path: "data/test-*" """

        # YAML Metadata for HF Hub
        yaml_header = f"""---
language:
- en
license: apache-2.0
task_categories:
- image-to-text
- object-detection
- text-generation
task_ids:
- image-captioning
- text-generation
pretty_name: {repo_name}
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
{splits_yaml}
---
"""

        body = f"""
# {repo_name}

{repo_name} is a high-quality document dataset generated by [DocParserEngine](https://github.com/remixonwin/data-parser).

## Dataset Summary
- **Documents Processed**: {num_docs}
- **Total Records**: {num_records}
- **Schema Format**: `{schema}`
- **Extraction Features**: Structural detection, image extraction, AI-powered captioning, and OCR.

## Supported Tasks
- **OCR & Text Extraction**: High-accuracy text extraction from complex document layouts.
- **Image Captioning & Categorization**: Vision-based descriptions and classification of extracted images.
- **RAG & Information Retrieval**: (When using `chunks` schema) Ideal for building knowledge bases.

## Dataset Structure
The dataset follows a standard `datasets.DatasetDict` structure with `train`, `validation`, and `test` splits.

### Schema: `{schema}`
Detailed feature descriptions can be found in the dataset configuration on the Hugging Face Hub.

## Maintenance
Generated and maintained using **DocParserEngine**.
"""
        return yaml_header + body

    def _create_splits(self, dataset):
        """Create train/validation splits."""
        import datasets

        n = len(dataset)
        if n < 4:
            return datasets.DatasetDict({"train": dataset})

        # 80/10/10 split
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        split = dataset.train_test_split(test_size=0.2, seed=42)
        val_test = split["test"].train_test_split(test_size=0.5, seed=42)

        return datasets.DatasetDict(
            {
                "train": split["train"],
                "validation": val_test["train"],
                "test": val_test["test"],
            }
        )
