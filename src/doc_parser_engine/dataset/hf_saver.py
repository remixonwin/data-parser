"""
HuggingFace Dataset Saver Module.

This module provides functionality to save extracted examples as HF-compliant
parquet shards with dataset_info.json.

Based on the design in plans/extraction_pipeline_design.md.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Default citation - should be customized per dataset
DEFAULT_CITATION = """@dataset{prepware_study_guide,
  author = {Aviation Supplies \& Academics, Inc.},
  title = {Prepware Study Guide},
  year = {2025},
  publisher = {ASA Online},
  url = {https://online.prepware.com/study_guide}
}"""

DEFAULT_DESCRIPTION = """Extracted from Prepware Study Guide documents. Each row represents one page from a source document.

The Prepware Study Guide is a comprehensive exam preparation resource for FAA Airframe and Powerplane (A&P) mechanic certification tests.

This dataset contains:
- Document metadata (title, authors, creation date)
- Per-page text content
- Extracted images with AI-generated captions
- Table data
- Extraction warnings and quality metrics"""


def get_hf_features() -> Dict[str, Any]:
    """Return the HuggingFace features definition matching the design spec.
    
    Returns:
        Dict containing the features definition in HF format
    """
    return {
        "example_id": {"dtype": "string", "_type": "Value"},
        "doc_id": {"dtype": "string", "_type": "Value"},
        "file_path": {"dtype": "string", "_type": "Value"},
        "file_name": {"dtype": "string", "_type": "Value"},
        "file_extension": {"dtype": "string", "_type": "Value"},
        "page_number": {"dtype": "int32", "_type": "Value"},
        "total_pages": {"dtype": "int32", "_type": "Value"},
        "doc_type": {"dtype": "string", "_type": "Value"},
        "title": {"dtype": "string", "_type": "Value"},
        "authors": {"feature": {"dtype": "string", "_type": "Value"}, "_type": "Sequence"},
        "created_at": {"dtype": "string", "_type": "Value"},
        "parsed_at": {"dtype": "string", "_type": "Value"},
        "extracted_text": {"dtype": "string", "_type": "Value"},
        "extracted_text_length": {"dtype": "int32", "_type": "Value"},
        "word_count": {"dtype": "int32", "_type": "Value"},
        "images": {
            "feature": {
                "image_id": {"dtype": "string", "_type": "Value"},
                "page": {"dtype": "int32", "_type": "Value"},
                "bbox": {"feature": {"dtype": "float32", "_type": "Value"}, "_type": "Sequence"},
                "width": {"dtype": "int32", "_type": "Value"},
                "height": {"dtype": "int32", "_type": "Value"},
                "format": {"dtype": "string", "_type": "Value"},
                "caption": {"dtype": "string", "_type": "Value"},
                "alt_text": {"dtype": "string", "_type": "Value"},
            },
            "_type": "Sequence"
        },
        "tables": {
            "feature": {
                "table_id": {"dtype": "string", "_type": "Value"},
                "page": {"dtype": "int32", "_type": "Value"},
                "bbox": {"feature": {"dtype": "float32", "_type": "Value"}, "_type": "Sequence"},
                "headers": {"feature": {"dtype": "string", "_type": "Value"}, "_type": "Sequence"},
                "rows": {"feature": {"feature": {"dtype": "string", "_type": "Value"}, "_type": "Sequence"}, "_type": "Sequence"},
                "caption": {"dtype": "string", "_type": "Value"},
            },
            "_type": "Sequence"
        },
        "metadata": {"dtype": "string", "_type": "Value"},
        "extraction_warnings": {"feature": {"dtype": "string", "_type": "Value"}, "_type": "Sequence"},
        "source_path": {"dtype": "string", "_type": "Value"},
        "file_hash": {"dtype": "string", "_type": "Value"},
    }


def create_dataset_info(
    citation: Optional[str] = None,
    description: Optional[str] = None,
    homepage: str = "https://online.prepware.com/study_guide",
    license: str = "proprietary"
) -> Dict[str, Any]:
    """Create the dataset_info.json structure.
    
    Args:
        citation: Citation string for the dataset
        description: Description of the dataset
        homepage: Dataset homepage URL
        license: License for the dataset
        
    Returns:
        Dict containing the complete dataset_info.json structure
    """
    return {
        "citation": citation or DEFAULT_CITATION,
        "description": description or DEFAULT_DESCRIPTION,
        "features": get_hf_features(),
        "homepage": homepage,
        "license": license,
        "annotations_creators": ["no-annotations"],
        "language": ["en"],
        "language_details": "English",
        "multilingual": False,
        "size_categories": ["n<1K", "1K<n<10K", "10K<n<100K"],
        "source_data": ["original"],
        "task_categories": ["feature-extraction"],
        "task_ids": ["extractive-qa", "table-to-text"]
    }


def transform_example_to_hf_format(example: Dict[str, Any]) -> Dict[str, Any]:
    """Transform an example to the HF-compatible format.
    
    The design spec expects these fields:
    - id, document_id, page_number, text, images[], source_path, file_hash, extraction_warnings
    
    But we use the full schema from the design:
    - example_id, doc_id, file_path, file_name, file_extension, page_number, total_pages
    - doc_type, title, authors, created_at, parsed_at
    - extracted_text, extracted_text_length, word_count
    - images[], tables[], metadata, extraction_warnings
    - source_path, file_hash
    
    Args:
        example: Raw example dict from extraction
        
    Returns:
        HF-compatible example dict
    """
    # Handle different possible input formats
    # The example might have different field names
    
    # Extract text - could be "text" or "extracted_text"
    text = example.get("extracted_text") or example.get("text", "")
    text_length = len(text)
    
    # Count words
    word_count = example.get("word_count") or (len(text.split()) if text else 0)
    
    # Handle images - could be list of dicts or other format
    images = example.get("images")
    if images is None:
        images = None  # Keep None for HF Sequence
    elif isinstance(images, list):
        if len(images) == 0:
            images = None  # Empty list should be None for HF Sequence
        else:
            # Normalize image format
            normalized_images = []
            for img in images:
                if isinstance(img, dict):
                    normalized_images.append({
                        "image_id": img.get("image_id", ""),
                        "page": img.get("page", 1),
                        "bbox": img.get("bbox", []) or [],
                        "width": img.get("width", 0),
                        "height": img.get("height", 0),
                        "format": img.get("format", ""),
                        "caption": img.get("caption", ""),
                        "alt_text": img.get("alt_text", ""),
                    })
            images = normalized_images
    
    # Handle tables - normalize format like images
    tables = example.get("tables")
    if tables is None:
        tables = None  # Keep None for HF Sequence
    elif isinstance(tables, list):
        if len(tables) == 0:
            tables = None  # Empty list should be None for HF Sequence
        else:
            # Normalize table format
            normalized_tables = []
            for tbl in tables:
                if isinstance(tbl, dict):
                    normalized_tables.append({
                        "table_id": tbl.get("table_id", ""),
                        "page": tbl.get("page", 1),
                        "bbox": tbl.get("bbox", []) or [],
                        "headers": tbl.get("headers", []) or [],
                        "rows": tbl.get("rows", []) or [],
                        "caption": tbl.get("caption", ""),
                    })
            tables = normalized_tables
    
    # Handle authors - empty list should be None
    authors = example.get("authors")
    if authors is None:
        authors = None
    elif isinstance(authors, list) and len(authors) == 0:
        authors = None
    
    # Handle extraction_warnings - empty list should be None
    extraction_warnings = example.get("extraction_warnings")
    if extraction_warnings is None:
        extraction_warnings = None
    elif isinstance(extraction_warnings, list) and len(extraction_warnings) == 0:
        extraction_warnings = None
    
    # Handle metadata - must be JSON string for HF
    metadata = example.get("metadata", {})
    if isinstance(metadata, dict):
        metadata = json.dumps(metadata)
    
    hf_example = {
        "example_id": example.get("example_id", ""),
        "doc_id": example.get("doc_id") or example.get("document_id", ""),
        "file_path": example.get("file_path") or example.get("source_path", ""),
        "file_name": example.get("file_name", Path(example.get("file_path", "")).name),
        "file_extension": example.get("file_extension", Path(example.get("file_path", "")).suffix),
        "page_number": example.get("page_number", 1),
        "total_pages": example.get("total_pages", example.get("page_count", 1)),
        "doc_type": example.get("doc_type", "pdf"),
        "title": example.get("title", ""),
        "authors": authors,  # Use converted authors (None for empty list)
        "created_at": example.get("created_at", ""),
        "parsed_at": example.get("parsed_at", ""),
        "extracted_text": text,
        "extracted_text_length": text_length,
        "word_count": word_count,
        "images": images,
        "tables": tables,
        "metadata": metadata,  # Use the converted metadata (JSON string)
        "extraction_warnings": extraction_warnings,  # Use converted extraction_warnings (None for empty list)
        "source_path": example.get("source_path") or example.get("file_path", ""),
        "file_hash": example.get("file_hash") or example.get("document_id", ""),
    }
    
    return hf_example


def save_hf_dataset(
    examples: List[Dict[str, Any]],
    output_dir: Path,
    split_name: str = "train",
    citation: Optional[str] = None,
    description: Optional[str] = None,
    shard_size: int = 10000
) -> List[Path]:
    """Save examples as HF-compliant parquet shards with dataset_info.json.
    
    Args:
        examples: List of example dicts
        output_dir: Output directory for the dataset
        split_name: Name of the split (train/validation/test)
        citation: Citation for dataset_info.json
        description: Description for dataset_info.json
        shard_size: Maximum records per shard
        
    Returns:
        List of created parquet file paths
    """
    try:
        from datasets import Dataset, Features, Value, Sequence
    except ImportError:
        raise RuntimeError("datasets library not installed. pip install datasets")
    
    # Transform examples to HF format
    hf_examples = [transform_example_to_hf_format(ex) for ex in examples]
    
    if not hf_examples:
        logger.warning("No examples to save!")
        return []
    
    # Create output directories
    data_dir = output_dir / "data" / split_name
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Define features
    features = Features({
        "example_id": Value("string"),
        "doc_id": Value("string"),
        "file_path": Value("string"),
        "file_name": Value("string"),
        "file_extension": Value("string"),
        "page_number": Value("int32"),
        "total_pages": Value("int32"),
        "doc_type": Value("string"),
        "title": Value("string"),
        "authors": Sequence(Value("string")),
        "created_at": Value("string"),
        "parsed_at": Value("string"),
        "extracted_text": Value("string"),
        "extracted_text_length": Value("int32"),
        "word_count": Value("int32"),
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
        "tables": Sequence({
            "table_id": Value("string"),
            "page": Value("int32"),
            "bbox": Sequence(Value("float32")),
            "headers": Sequence(Value("string")),
            "rows": Sequence(Sequence(Value("string"))),
            "caption": Value("string"),
        }),
        "metadata": Value("string"),
        "extraction_warnings": Sequence(Value("string")),
        "source_path": Value("string"),
        "file_hash": Value("string"),
    })
    
    # Create dataset
    dataset = Dataset.from_list(hf_examples, features=features)
    
    # Save as parquet shards
    parquet_files = []
    num_shards = (len(hf_examples) + shard_size - 1) // shard_size
    
    for i in range(num_shards):
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, len(hf_examples))
        
        shard = dataset.select(range(start_idx, end_idx))
        shard_filename = f"data-{i:05d}-of-{num_shards:05d}.parquet"
        shard_path = data_dir / shard_filename
        
        shard.to_parquet(str(shard_path))
        parquet_files.append(shard_path)
        
        logger.info(f"Wrote shard {i+1}/{num_shards}: {shard_path}")
    
    # Write dataset_info.json in the split directory (HF standard)
    dataset_info = create_dataset_info(citation=citation, description=description)
    dataset_info_path = data_dir / "dataset_info.json"
    
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Wrote dataset_info.json to {dataset_info_path}")
    
    # Also write dataset_info.json at the root level for load_dataset compatibility
    root_dataset_info_path = output_dir / "data" / "dataset_info.json"
    with open(root_dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Wrote dataset_info.json to {root_dataset_info_path}")
    
    # Write dataset_dict.json for load_dataset compatibility
    dataset_dict = {
        "splits": {
            split_name: {
                "num_bytes": sum(f.stat().st_size for f in parquet_files),
                "num_examples": len(hf_examples)
            }
        }
    }
    dataset_dict_path = output_dir / "data" / "dataset_dict.json"
    
    with open(dataset_dict_path, "w", encoding="utf-8") as f:
        json.dump(dataset_dict, f, indent=2)
    
    logger.info(f"Wrote dataset_dict.json to {dataset_dict_path}")
    
    return parquet_files


def verify_hf_dataset_load(dataset_path: Path) -> Dict[str, Any]:
    """Verify that the local dataset can be loaded via datasets.load_from_disk or load_dataset.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        Dict with verification results: {
            "success": bool,
            "load_method": str,
            "splits": list,
            "num_examples": int,
            "features": list,
            "error": str or None
        }
    """
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError:
        return {
            "success": False,
            "load_method": None,
            "splits": [],
            "num_examples": 0,
            "features": [],
            "error": "datasets library not installed"
        }
    
    data_dir = dataset_path / "data"
    
    # Try load_from_disk first
    try:
        logger.info(f"Trying load_from_disk: {data_dir}")
        ds = load_from_disk(str(data_dir))
        
        # Get info
        splits = list(ds.keys()) if hasattr(ds, 'keys') else [ds.split]
        num_examples = sum(len(ds[s]) for s in splits) if hasattr(ds, '__getitem__') else len(ds)
        features = list(ds.features.keys()) if hasattr(ds, 'features') else []
        
        return {
            "success": True,
            "load_method": "load_from_disk",
            "splits": splits,
            "num_examples": num_examples,
            "features": features,
            "error": None
        }
    except Exception as e:
        logger.info(f"load_from_disk failed: {e}")
    
    # Try load_dataset with filesystem
    try:
        logger.info(f"Trying load_dataset from filesystem: {data_dir}")
        ds = load_dataset(str(data_dir), split="train")
        
        splits = ["train"]
        num_examples = len(ds)
        features = list(ds.features.keys()) if hasattr(ds, 'features') else []
        
        return {
            "success": True,
            "load_method": "load_dataset (filesystem)",
            "splits": splits,
            "num_examples": num_examples,
            "features": features,
            "error": None
        }
    except Exception as e:
        logger.info(f"load_dataset failed: {e}")
        return {
            "success": False,
            "load_method": None,
            "splits": [],
            "num_examples": 0,
            "features": [],
            "error": str(e)
        }
