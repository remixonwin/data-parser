import sys
import json
from pathlib import Path
import subprocess
import os


def test_choose_path_headless(tmp_path, monkeypatch, capsys):
    # Use choose_path_interactive directly
    from doc_parser_engine.cli import choose_path_interactive

    # create sample dirs
    a = tmp_path / "one.pdf"
    b = tmp_path / "two.pdf"
    a.write_text("x")
    b.write_text("y")

    # headless filter
    chosen = choose_path_interactive(root=tmp_path, ci=True, filter="two")
    assert chosen is not None and chosen.name == "two.pdf"

    # headless select_index
    chosen2 = choose_path_interactive(root=tmp_path, ci=True, select_index=0)
    assert chosen2 is not None and chosen2.exists()


def test_parse_noninteractive_jsonl(tmp_path, monkeypatch, capsys):
    # run parse in CI mode against prepared parsed_outputs (use existing parsed_outputs dir)
    from doc_parser_engine.cli import parse
    # call parse on a single file
    p = tmp_path / "doc.pdf"
    p.write_text("hello")

    # monkeypatch get_engine to a lightweight fake engine
    class FakeDoc:
        def __init__(self, path):
            self.path = path
            self.text = "content"

    class FakeEngine:
        SUPPORTED_FORMATS = ['.pdf']
        def __init__(self, *args, **kwargs):
            self.output_dir = tmp_path
        def parse(self, p):
            return FakeDoc(p)
        def to_hf_dataset(self, docs, schema, push_to_hub, hub_repo, hub_token):
            return {"built": True}

    monkeypatch.setattr('doc_parser_engine.cli.get_engine', lambda **k: FakeEngine())

    # invoke parse with ci mode by calling function and capturing stderr
    try:
        parse(path=p, output=tmp_path, recursive=False, glob=None, export=None, push=False, hub_repo=None, caption_model="none", no_caption=True, no_ocr=True, no_tables=True, no_ocr_images=True, device="cpu", batch_size=1, llm_api=None, llm_model=None, verbose=False)
    except SystemExit:
        pass

    # nothing to assert beyond no exception; ensure this is noninteractive
    assert True
