import os
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("JOB_CHROMA_PERSIST_DIR", "/tmp/resumetooffer_test_chroma")
