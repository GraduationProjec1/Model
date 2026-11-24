import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
PDP_DIR = REPO_ROOT / "pdp"

for path in (REPO_ROOT, PDP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
