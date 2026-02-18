import os
import sys
import logging
import re
from pathlib import Path
from typing import Optional, List
import json

import typer
try:
    import questionary
except Exception:
    questionary = None
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

DEFAULT_CAPTION_MODEL = "Salesforce/blip2-opt-2.7b"


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


def _choose_caption_model(
    llm_api_base: Optional[str],
    llm_model: Optional[str],
    explicit_caption_model: Optional[str],
    force_local: bool,
) -> str:
    """Choose the final caption model with auto-selection rules.

    Rules implemented:
      - If an explicit caption_model was provided, honor it only if it's 'api' or force_local is True.
      - If explicit_caption_model is provided but not force_local and it's not 'api', and a remote LLM
        (llm_api_base + llm_model) is available, prefer 'api'.
      - If no explicit model and llm_api_base and llm_model are provided and not force_local -> 'api'.
      - Otherwise return the default HF model.
    """
    logger = logging.getLogger(__name__)

    if explicit_caption_model:
        if explicit_caption_model == "api":
            return "api"
        if force_local:
            return explicit_caption_model
        # explicit HF model provided but user didn't force local; prefer API if remote LLM exists
        if llm_api_base and llm_model:
            logger.info("Auto-selecting 'api' caption backend because LLM API is available and --force-local-caption not set")
            return "api"
        return explicit_caption_model

    # No explicit model
    if llm_api_base and llm_model and not force_local:
        logger.info(f"Using remote LLM API at {llm_api_base} with model {llm_model}; captioning backend set to 'api' (no large HF downloads)")
        return "api"

    return DEFAULT_CAPTION_MODEL


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
    force_local_caption: bool = False,
    verbose: bool = False,
) -> DocParserEngine:
    """Utility to initialize the DocParserEngine with consistent settings.

    Precedence for LLM settings: CLI > ENV > Config file.
    Environment variables supported: DOCPARSER_LLM_API_BASE, DOCPARSER_LLM_MODEL
    """
    config = load_config()

    # Env overrides
    env_llm_api = os.getenv("DOCPARSER_LLM_API_BASE")
    env_llm_model = os.getenv("DOCPARSER_LLM_MODEL")

    # Determine if a caption model was explicitly provided.
    # Preserve CLI flag default behavior: parse() still provides DEFAULT_CAPTION_MODEL by default,
    # so treat that value as "not explicitly set" unless the config provides a caption_model.
    config_caption = config.get("caption_model")
    explicit_caption_model = None
    if caption_model is not None and caption_model != DEFAULT_CAPTION_MODEL:
        # User provided a non-default model via CLI
        explicit_caption_model = caption_model
    elif config_caption:
        # User provided a model via config
        explicit_caption_model = config_caption

    # Effective LLM settings: CLI > ENV > Config
    effective_llm_api = llm_api_base or env_llm_api or config.get("llm_api_base")
    effective_llm_model = llm_model or env_llm_model or config.get("llm_model")

    final_model = _choose_caption_model(effective_llm_api, effective_llm_model, explicit_caption_model, force_local_caption)

    # Prioritize other settings: CLI args > Config File > Defaults
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
        llm_api_base=effective_llm_api,
        llm_model=effective_llm_model or "gpt-4o",
        force_local_caption=force_local_caption,
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
        default=current.get("caption_model", DEFAULT_CAPTION_MODEL)
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
    caption_model: str = typer.Option(DEFAULT_CAPTION_MODEL, "--model", help="Captioning model"),
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
                    "3. Update [bold].env[/bold] with the correct [bold]HF_TOKEN[/bold]..",
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
        caption_model="none" if no_caption else DEFAULT_CAPTION_MODEL,
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


def choose_path_interactive(root: Path = Path.cwd(), input_func=input, print_func=print, max_entries: int = 200) -> Optional[Path]:
    """Interactive path chooser rooted at `root`.

    Returns the chosen Path or None if cancelled.
    """
    workspace_root = Path(root).resolve()
    current = workspace_root
    filter_str = ""

    def _list_entries(directory: Path) -> List[Path]:
        try:
            entries = [p for p in directory.iterdir()]
        except Exception:
            return []
        # Sort: directories first, then files, both alphabetically
        entries.sort(key=lambda p: (p.is_file(), p.name.lower()))
        return entries

    while True:
        all_entries = _list_entries(current)
        if filter_str:
            entries = [p for p in all_entries if filter_str.lower() in p.name.lower()]
        else:
            entries = all_entries

        entries = entries[:max_entries]

        print_func(f"\nCurrent: {str(current.relative_to(workspace_root) if current != workspace_root else Path('.'))}")
        if not entries:
            print_func("(No entries)")
        for idx, p in enumerate(entries, start=1):
            rel = p.relative_to(workspace_root) if workspace_root in p.parents or p == workspace_root else p
            kind = "[DIR]" if p.is_dir() else "[FILE]"
            print_func(f"{idx:3d}. {kind} {rel}")

        prompt = "Enter filter (substring), a number to select, 'r <n>' to recurse into a directory, 'u' to go up, or 'q' to cancel: "
        resp = input_func(prompt).strip()

        if resp == "q":
            return None
        if resp == "u":
            if current == workspace_root:
                print_func("Already at workspace root")
            else:
                current = current.parent
                filter_str = ""
            continue

        if resp.startswith("r"):
            rest = resp[1:].strip()
            if not rest:
                # ask for number
                rest = input_func("Enter directory number to recurse into: ").strip()
            if rest.isdigit():
                idx = int(rest) - 1
                if 0 <= idx < len(entries):
                    target = entries[idx]
                    if target.is_dir():
                        current = target
                        filter_str = ""
                    else:
                        print_func("Selected entry is not a directory")
                else:
                    print_func("Index out of range")
            else:
                print_func("Invalid directory index")
            continue

        if resp.isdigit():
            idx = int(resp) - 1
            if 0 <= idx < len(entries):
                return entries[idx].resolve()
            else:
                print_func("Index out of range")
                continue

        # Otherwise treat as filter
        filter_str = resp
        continue


@app.command("choose-path")
def choose_path_cmd():
    """Interactive chooser to select a file or directory under the current workspace.

    Prints the absolute path of the selected entry to stdout and exits with code 0.
    If the user cancels, exits with code 2.
    """
    chosen = choose_path_interactive(Path.cwd(), input_func=input, print_func=print)
    if chosen:
        # Print absolute path
        print(str(chosen))
        raise typer.Exit(0)
    else:
        raise typer.Exit(2)


@app.command("audit-dataset")
def audit_dataset(
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="HF repo id"),
    local_src: Path = typer.Option(Path("prepware_study_guide"), "--local-src", help="Local source dir"),
    sample: int = typer.Option(10, "--sample", "-s", help="Sample size"),
    threshold: float = typer.Option(0.6, "--threshold", help="Similarity threshold"),
    dest: Optional[Path] = typer.Option(None, "--dest", help="Output directory"),
    no_download: bool = typer.Option(False, "--no-download", help="Skip downloading HF artifacts"),
):
    """Run an automated audit of a HuggingFace dataset by sampling records and comparing to local sources.

    Exit codes: 0 success, 1 verification failures, 2 usage/errors.
    """
    import datetime
    from doc_parser_engine import audit as audit_mod

    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(dest) if dest else Path("parsed_outputs") / "prepware_audit" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_id = repo or os.getenv("REPO_ID") or get_default_repo_id(Path.cwd(), suggested_title=None)
    if not repo_id:
        rprint("[red]No repository id provided and REPO_ID env not set[/red]")
        raise typer.Exit(2)

    try:
        if not no_download:
            rprint(f"[dim]Downloading HF artifacts for {repo_id} into {out_dir}[/dim]")
            art = audit_mod.download_hf_artifacts(repo_id, out_dir)
        else:
            art = {"downloaded_files": [], "metadata_path": None}

        results = audit_mod.sample_and_compare(repo_id=repo_id, local_src=local_src, dest=out_dir, sample_size=sample, similarity_threshold=threshold)
        json_path, md_path = audit_mod.generate_report(results, out_dir)
        rprint(f"[green]Audit completed. JSON: {json_path} MD: {md_path}[/green]")

        failed = any(not s.get("pass") for s in results.get("samples", []))
        if failed:
            rprint(f"[yellow]Some samples failed audit (pass_rate={results['summary']['pass_rate']:.2%}). Exiting with code 1[/yellow]")
            raise typer.Exit(1)
        raise typer.Exit(0)
    except typer.Exit:
        raise
    except Exception as e:
        rprint(f"[red]Audit failed: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
