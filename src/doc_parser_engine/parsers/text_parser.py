import re
from pathlib import Path

class TextParser:
    def __init__(self, **kwargs):
        pass

    def parse(self, path):
        path = Path(path)
        content = path.read_text(encoding="utf-8")
        
        elements = []
        raw_text = content
        
        lines = content.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
                
            # Markdown headings
            match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if match:
                level = len(match.group(1))
                elements.append({
                    "type": f"heading_{level}",
                    "content": match.group(2).strip(),
                    "page": 0,
                    "bbox": [0, 0, 0, 0],
                })
                i += 1
                continue
            
            # Underline headings
            if i + 1 < len(lines):
                next_line = lines[i+1].strip()
                if re.match(r'^={3,}$', next_line):
                    elements.append({
                        "type": "heading_1",
                        "content": line,
                        "page": 0,
                        "bbox": [0, 0, 0, 0],
                    })
                    i += 2
                    continue
                elif re.match(r'^-{3,}$', next_line):
                    elements.append({
                        "type": "heading_2",
                        "content": line,
                        "page": 0,
                        "bbox": [0, 0, 0, 0],
                    })
                    i += 2
                    continue
            
            # Paragraphs
            elements.append({
                "type": "paragraph",
                "content": line,
                "page": 0,
                "bbox": [0, 0, 0, 0],
            })
            i += 1
            
        return {
            "raw_elements": elements,
            "raw_text": raw_text,
            "title": path.name,
            "doc_type": path.suffix[1:] if path.suffix else "txt"
        }
