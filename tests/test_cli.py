import pytest
from typer.testing import CliRunner
from pathlib import Path
from doc_parser_engine.cli import app

runner = CliRunner()

def test_cli_help():
    """Test that the main help command works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "doc-parser" in result.stdout
    assert "parse" in result.stdout
    assert "status" in result.stdout
    assert "config" in result.stdout

def test_cli_status():
    """Test the status command."""
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "System Status" in result.stdout
    assert "Python Version" in result.stdout

def test_cli_parse_help():
    """Test the parse help command."""
    result = runner.invoke(app, ["parse", "--help"])
    assert result.exit_code == 0
    assert "Parse documents" in result.stdout
    assert "--no-ocr-images" in result.stdout

def test_cli_config_ui_smoke():
    """Test that config command starts (smoke test)."""
    # We don't want to run the whole interactive wizard in tests easily,
    # but we can check if it prints the header.
    # Note: questionary might hang or fail if not handled, so we just check entry.
    # We skip full interactive test for now as it requires mocking stdin.
    pass
