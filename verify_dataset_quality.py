import logging
try:
    from doc_parser_engine.core import DocParserEngine
except ModuleNotFoundError:
    # Allow running directly with PYTHONPATH=. by adding src/ to sys.path when needed
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from doc_parser_engine.core import DocParserEngine
from pathlib import Path
from rich.console import Console
from rich.table import Table

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO)

console = Console()
# Use the API for verification as requested
engine = DocParserEngine(
    enable_captioning=True,
    caption_model="api",
    llm_api_base="http://0.0.0.0:7543",
    llm_model="gpt-4o",
    verbose=True
)
doc_path = Path("wisconsin.pdf")

if not doc_path.exists():
    console.print(f"[yellow]Warning: {doc_path} not found. Skipping quality verification in this environment.[/yellow]")
    # Exit successfully to allow CI to continue when sample document is not present
    exit(0)

console.print(f"[bold blue]Parsing {doc_path} via API...[/bold blue]")
doc = engine.parse(doc_path)

console.print(f"\n[bold green]Extraction Summary for {doc.title}[/bold green]")
console.print(f"Total Sections: {len(doc.sections)}")
console.print(f"Total Paragraphs: {len(doc.paragraphs)}")
console.print(f"Total Images: {len(doc.images)}")
console.print(f"Total Tables: {len(doc.tables)}")

if doc.images:
    table = Table(title="Image Extraction & API Caption Quality")
    table.add_column("ID", style="cyan")
    table.add_row("Page", style="magenta")
    table.add_column("Category", style="yellow")
    table.add_column("Caption (LLM API)", style="green")
    table.add_column("OCR Hint", style="dim blue")

    for img in doc.images[:15]: # Check first 15
        table.add_row(
            img.get("image_id", "N/A")[:8],
            str(img.get("page", "N/A")),
            img.get("category", "N/A"),
            img.get("caption", "MISSING"),
            (img.get("ocr_text", "")[:30] + "...") if img.get("ocr_text") else "None"
        )
    console.print(table)
else:
    console.print("[yellow]No images extracted.[/yellow]")

# Check text sample from various parts
if doc.sections:
    console.print("\n[bold]Text Quality Samples:[/bold]")
    for i, section in enumerate(doc.sections[:3]):
        console.print(f"\n[cyan]Section {i+1}: {section.get('title', 'Untitled')}[/cyan]")
        console.print(section.get("content", "")[:300] + "...")

if doc.paragraphs:
    console.print(f"\n[bold]Total word count:[/bold] {doc.word_count}")
