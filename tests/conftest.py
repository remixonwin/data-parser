import sys
from pathlib import Path

# Ensure src/ is on sys.path so packages under src can be imported as top-level modules
ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
