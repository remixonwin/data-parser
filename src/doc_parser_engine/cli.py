import os
import sys
import logging
import re
from pathlib import Path
from typing import Optional, List
import json

import typer
import questionary
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich import print as rprint

from .core import DocParserEngine, ParsedDocument

from dotenv import load_dotenv
load_dotenv()

app = typer.Typer(
    name="doc-parser",
    help="🚀 Production-grade document parsing engine with HuggingFace integration.",
    add_completion=False,
)
console = Console()
CONFIG_PATH = Path.home() / ".doc-parser-config.json"

def load_config() -> dict:
    """Load user configuration from disk."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_config(config: dict):
    """Save user configuration to disk."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)

def get_engine(
    caption_model: Optional[str] = None,
    no_caption: bool = False,
    no_ocr: bool = False,
    no_tables: bool = False,
    no_ocr_images: bool = False,
    output_dir: Optional[str] = None,
    device: str = "auto",
    batch_size: int = 8,
    llm_api_base: Optional[str] = None,
    llm_model: Optional[str] = None,
    verbose: bool = False,
) -> DocParserEngine:
    """Utility to initialize the DocParserEngine with consistent settings."""
    config = load_config()
    
    # Prioritize CLI args > Config File > Defaults
    final_model = caption_model or config.get("caption_model", "Salesforce/blip2-opt-2.7b")
    final_device = device if device != "auto" else config.get("device", "auto")
    final_output = output_dir or config.get("output_dir")
    final_ocr_images = not no_ocr_images if no_ocr_images else config.get("enable_ocr_on_images", True)
    
    return DocParserEngine(
        caption_model=final_model,
        enable_captioning=not no_caption,
        enable_ocr=not no_ocr,
        enable_table_extraction=not no_tables,
        enable_ocr_on_images=final_ocr_images,
        output_dir=final_output,
        device=final_device,
        batch_size=batch_size,
        llm_api_base=llm_api_base or config.get("llm_api_base"),
        llm_model=llm_model or config.get("llm_model", "gpt-4o"),
        verbose=verbose,
    )

def get_default_repo_id(path: Path, suggested_title: Optional[str] = None) -> Optional[str]:
    """Generate an intelligent repository ID based on title and environment."""
    username = os.getenv("HF_USERNAME")
    if not username:
        return None
    
    name_source = suggested_title or (path.stem if path.is_file() else path.name)
    repo_slug = re.sub(r'[^a-zA-Z0-9\-_]', '-', name_source).lower()
    repo_slug = re.sub(r'-+', '-', repo_slug).strip('-')
    return f"{username}/{repo_slug}-dataset"

@app.command()
def status():
    """Check system compatibility and available hardware."""
    table = Table(title="DocParserEngine System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    # Check Python version
    table.add_row("Python Version", sys.version.split()[0])
    
    # Check Torch/Device
    try:
        import torch
        device = "cpu"
        if torch.cuda.is_available():
            device = f"cuda ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        table.add_row("Compute Device", device)
    except ImportError:
        table.add_row("Compute Device", "[red]torch not installed[/red]")

    # Check Tesseract
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        table.add_row("OCR (Tesseract)", "Available")
    except Exception:
        table.add_row("OCR (Tesseract)", "[yellow]Not found (Install tesseract-ocr for OCR features)[/yellow]")

    console.print(table)

@app.command()
def config():
    """Interactive configuration of default engine parameters."""
    current = load_config()
    
    rprint("[bold blue]DocParserEngine Configuration Wizard[/bold blue]")
    rprint("[dim]These settings will be saved to ~/.doc-parser-config.json[/dim]\n")
    
    model = questionary.text(
        "Default Caption Model ID:",
        default=current.get("caption_model", "Salesforce/blip2-opt-2.7b")
    ).ask()
    
    device = questionary.select(
        "Default Compute Device:",
        choices=["auto", "cpu", "cuda", "mps"],
        default=current.get("device", "auto")
    ).ask()
    
    output_dir = questionary.text(
        "Default Output Directory:",
        default=current.get("output_dir", "./parsed_outputs")
    ).ask()
    
    ocr_images = questionary.confirm(
        "Enable OCR on images by default?",
        default=current.get("enable_ocr_on_images", True)
    ).ask()

    llm_api = questionary.text(
        "LLM API Base URL (optional):",
        default=current.get("llm_api_base", "")
    ).ask()

    llm_model = questionary.text(
        "LLM Model ID:",
        default=current.get("llm_model", "gpt-4o")
    ).ask()
    
    new_config = {
        "caption_model": model,
        "device": device,
        "output_dir": output_dir,
        "enable_ocr_on_images": ocr_images,
        "llm_api_base": llm_api if llm_api else None,
        "llm_model": llm_model
    }
    
    save_config(new_config)
    rprint("\n[bold green]✓ Configuration saved successfully![/bold green]")

@app.command()
def parse(
    path: Path = typer.Argument(..., help="File or directory to parse"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recurse into subdirectories"),
    glob: Optional[str] = typer.Option(None, "--glob", help="Glob pattern for directory search"),
    export: str = typer.Option("full", "--export", help="HF Dataset schema (full, chunks, images, minimal, qa)"),
    push: bool = typer.Option(False, "--push", help="Push to HuggingFace Hub"),
    hub_repo: Optional[str] = typer.Option(None, "--hub-repo", help="HF Hub repository ID"),
    caption_model: str = typer.Option("Salesforce/blip2-opt-2.7b", "--model", help="Captioning model"),
    no_caption: bool = typer.Option(False, "--no-caption", help="Disable image captioning"),
    no_ocr: bool = typer.Option(False, "--no-ocr", help="Disable OCR"),
    no_tables: bool = typer.Option(False, "--no-tables", help="Disable table extraction"),
    no_ocr_images: bool = typer.Option(False, "--no-ocr-images", help="Disable OCR on images"),
    device: str = typer.Option("auto", "--device", help="Device (cpu, cuda, mps, auto)"),
    batch_size: int = typer.Option(8, "--batch-size", help="Batch size for captioning"),
    llm_api: Optional[str] = typer.Option(None, "--llm-api", help="LLM API Base URL"),
    llm_model: Optional[str] = typer.Option(None, "--llm-model", help="LLM Model ID"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Parse documents and optionally export to HuggingFace Datasets."""
    engine = get_engine(
        caption_model=caption_model,
        no_caption=no_caption,
        no_ocr=no_ocr,
        no_tables=no_tables,
        no_ocr_images=no_ocr_images,
        output_dir=str(output) if output else None,
        device=device,
        batch_size=batch_size,
        llm_api_base=llm_api,
        llm_model=llm_model,
        verbose=verbose,
    )

    if path.is_file():
        files = [path]
    elif path.is_dir():
        if glob:
            files = list(path.glob(glob))
        elif recursive:
            files = [f for ext in engine.SUPPORTED_FORMATS for f in path.rglob(f"*{ext}")]
        else:
            files = [f for ext in engine.SUPPORTED_FORMATS for f in path.glob(f"*{ext}")]
        files = sorted(set(files))
    else:
        rprint(f"[red]Error:[/red] Path {path} does not exist.")
        raise typer.Exit(1)

    if not files:
        rprint("[yellow]No supported files found.[/yellow]")
        return

    docs = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Parsing documents...", total=len(files))
        
        for file_path in files:
            progress.update(task, description=f"[cyan]Parsing {file_path.name}...")
            try:
                doc = engine.parse(file_path)
                docs.append(doc)
            except Exception as e:
                console.print(f"[red]Failed to parse {file_path.name}: {e}[/red]")
            progress.advance(task)

    if not docs:
        rprint("[red]No documents were successfully parsed.[/red]")
        return

    rprint(f"\n[green]Successfully parsed {len(docs)} documents.[/green]")
    
    # Export to Dataset
    if export:
        hub_token = os.getenv("HF_TOKEN")
        
        # Auto-generate hub_repo if pushing but not provided
        if push and not hub_repo:
            hub_repo = get_default_repo_id(path)
            if hub_repo:
                rprint(f"[dim]Automatically using intelligent repo ID: {hub_repo}[/dim]")

        try:
            with console.status(f"[bold blue]Building {export} dataset..."):
                dataset = engine.to_hf_dataset(
                    docs, 
                    schema=export,
                    push_to_hub=push,
                    hub_repo=hub_repo,
                    hub_token=hub_token
                )
            rprint(f"[bold green]Dataset built successfully![/bold green] (Schema: {export})")
        except Exception as e:
            msg = str(e)
            if "403 Forbidden" in msg or "rights to create a dataset" in msg:
                rprint("[bold red]Error: Permission Denied (403 Forbidden)[/bold red]")
                rprint(Panel(
                    "Your HuggingFace token does not have [bold]Write/Create[/bold] permissions.\n\n"
                    "1. Visit [cyan]https://huggingface.co/settings/tokens[/cyan]\n"
                    "2. Ensure your token type is [bold]Write[/bold] or has 'Create Repo' scopes.\n"
                    "3. Update [bold].env[/bold] with the correct [bold]HF_TOKEN[/bold].",
                    title="Permissions Required",
                    border_style="red"
                ))
            else:
                rprint(f"[red]Error during export: {e}[/red]")
            
        if not push:
            rprint("[dim]Use --push and --hub-repo to share on HuggingFace Hub.[/dim]")

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Interactive wizard if no command is provided."""
    if ctx.invoked_subcommand is not None:
        return

    rprint(Panel.fit(
        "[bold cyan]DocParserEngine[/bold cyan] 🔍📄\n"
        "[dim]Interactive Document Processing Wizard[/dim]",
        border_style="cyan"
    ))

    choice = questionary.select(
        "What would you like to do?",
        choices=[
            "Parse a document or directory",
            "Check system status",
            "Exit"
        ]
    ).ask()

    if choice == "Check system status":
        status()
    elif choice == "Parse a document or directory":
        interactive_parse()
    else:
        raise typer.Exit()

def interactive_parse():
    path_str = questionary.path("Enter file or directory path:").ask()
    if not path_str:
        return
    path = Path(path_str)
    
    recursive = False
    if path.is_dir():
        recursive = questionary.confirm("Recurse into subdirectories?", default=True).ask()

    export = questionary.select(
        "Choose export schema:",
        choices=["full", "chunks", "images", "minimal", "qa"]
    ).ask()

    push = questionary.confirm("Push to HuggingFace Hub?", default=False).ask()
    hub_repo = None
    if push:
        # Scout for a better title if it's a single file
        suggested_title = None
        if path.is_file():
            with console.status("[bold blue]Scouting document title..."):
                # Use a lightweight engine just for scouting
                temp_engine = DocParserEngine()
                metadata = temp_engine.scout_metadata(path)
                suggested_title = metadata.get("title")

        default_repo = get_default_repo_id(path, suggested_title=suggested_title)
        
        # If we have a default repo, we use it without asking to confirm
        # to streamline the experience as requested by the user.
        if default_repo:
            hub_repo = default_repo
            rprint(f"[dim]Intelligently selected repo: {hub_repo}[/dim]")
        else:
            hub_repo = questionary.text(
                "Enter Hub Repository ID (e.g. username/dataset):"
            ).ask()

    # Advanced options
    no_tables = False
    no_ocr_images = False
    if questionary.confirm("Configure advanced options?", default=False).ask():
        no_caption = not questionary.confirm("Enable image captioning?", default=True).ask()
        no_ocr = not questionary.confirm("Enable OCR?", default=True).ask()
        no_tables = not questionary.confirm("Enable table extraction?", default=True).ask()
        no_ocr_images = not questionary.confirm("Enable OCR on images for categorization?", default=True).ask()
        device = questionary.select("Device:", choices=["auto", "cpu", "cuda", "mps"]).ask()
        llm_api = questionary.text("LLM API Base URL (empty to disable):", default="http://0.0.0.0:7543").ask()
        llm_model = questionary.text("LLM Model ID:", default="gpt-4o").ask()
    else:
        no_caption = False
        no_ocr = False
        device = "auto"
        llm_api = None
        llm_model = None

    engine = get_engine(
        no_caption=no_caption,
        no_ocr=no_ocr,
        no_tables=no_tables,
        no_ocr_images=no_ocr_images,
        device=device,
        llm_api_base=llm_api,
        llm_model=llm_model,
        verbose=True  # Default verbose for interactive
    )

    # Invoke the parse command
    parse(
        path=path,
        output=None,
        recursive=recursive,
        glob=None,
        export=export,
        push=push,
        hub_repo=hub_repo,
        caption_model="none" if no_caption else "Salesforce/blip2-opt-2.7b",
        no_caption=no_caption,
        no_ocr=no_ocr,
        no_tables=no_tables,
        no_ocr_images=no_ocr_images,
        device=device,
        batch_size=8,
        llm_api=llm_api,
        llm_model=llm_model,
        verbose=True
    )

if __name__ == "__main__":
    app()
