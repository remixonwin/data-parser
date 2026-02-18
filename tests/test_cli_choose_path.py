import os
from pathlib import Path
from src.doc_parser_engine.cli import choose_path_interactive


def test_choose_path_filter_and_select(tmp_path, monkeypatch, capsys):
    # Setup a fake directory structure
    root = tmp_path
    (root / "dirA").mkdir()
    (root / "dirB").mkdir()
    (root / "file1.txt").write_text("hello")
    (root / "file2.log").write_text("log")

    inputs = ["", "file1", "1"]  # first show, then filter by 'file1', then select first
    def fake_input(prompt=""):
        return inputs.pop(0)

    chosen = choose_path_interactive(root, input_func=fake_input, print_func=lambda *a, **k: None)
    assert chosen is not None
    assert chosen == (root / "file1.txt").resolve()


def test_choose_path_cancel(tmp_path, monkeypatch):
    root = tmp_path
    (root / "a").mkdir()
    inputs = ["q"]
    def fake_input(prompt=""):
        return inputs.pop(0)
    chosen = choose_path_interactive(root, input_func=fake_input, print_func=lambda *a, **k: None)
    assert chosen is None
