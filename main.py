import sys
from pathlib import Path

# Add src to sys.path to allow importing the package locally
sys.path.insert(0, str(Path(__file__).parent / "src"))

from doc_parser_engine import DocParserEngine


def main():
    print("🚀 DocParserEngine - Quick Start")
    
    # Initialize the engine
    engine = DocParserEngine(
        enable_captioning=False,  # Set to True if ML deps are installed
        enable_ocr=False,         # Set to True if Tesseract is installed
        verbose=True
    )
    
    # Example: How to used the engine
    print("\nUsage Example:")
    print("1. Create an engine instance.")
    print("2. Parse a document: doc = engine.parse('path/to/doc.pdf')")
    print("3. Export to dataset: dataset = engine.to_hf_dataset([doc])")
    
    print("\nReady to parse documents!")


if __name__ == "__main__":
    main()
