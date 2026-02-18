from pathlib import Path
from doc_parser_engine.detection.structure_detector import detect_input_types
import tempfile

def test_detect_text_only(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("hello")
    assert detect_input_types([str(f)]) == 'text-only'

def test_detect_image_only(tmp_path):
    f = tmp_path / "img.jpg"
    f.write_bytes(b"\xff\xd8\xff")
    assert detect_input_types([str(f)]) == 'image-only'

def test_detect_combined(tmp_path):
    t = tmp_path / "doc.txt"
    i = tmp_path / "img.png"
    t.write_text("hi")
    i.write_bytes(b"\x89PNG\r\n")
    assert detect_input_types([str(t), str(i)]) == 'combined'

def test_detect_dir_mixed(tmp_path):
    d = tmp_path / "dir"
    d.mkdir()
    (d / "a.md").write_text("hello")
    (d / "b.jpg").write_bytes(b"jpeg")
    assert detect_input_types([str(d)]) == 'combined'
