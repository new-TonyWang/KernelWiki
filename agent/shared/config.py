"""Environment-based config for the KB-gen agent."""

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get(
    "KP_PROJECT_ROOT",
    str(Path(__file__).resolve().parent.parent.parent),
))
KNOWLEDGE_ROOT = PROJECT_ROOT / "knowledge"
SOURCE_CORPUS_ROOT = KNOWLEDGE_ROOT / "05-source-corpus"
SOURCE_CORPUS_MANIFEST = SOURCE_CORPUS_ROOT / "MANIFEST.yaml"
SOURCE_CORPUS_INDEX_ROOT = SOURCE_CORPUS_ROOT / "INDEX"

REMOTE_HOST = os.environ.get("KP_REMOTE_HOST", "h200_ncu")
REMOTE_DIR = os.environ.get(
    "KP_REMOTE_DIR",
    "/inspire/hdd/project/qianghuaxuexi/public/wty/ai4ai/ai-infra/kernel-kb-mvp",
)
REMOTE_CUDA_PATH = os.environ.get("KP_REMOTE_CUDA_PATH", "/usr/local/cuda-12.9/bin")
REMOTE_PYLIB = os.environ.get("KP_REMOTE_PYLIB", "/tmp/pylib")

OPENAI_BASE_URL = os.environ.get("KP_OPENAI_BASE_URL", "http://localhost:4000")
OPENAI_API_KEY = os.environ.get("KP_OPENAI_API_KEY", "sk-RBMjIVcX5LcKtnYZKNm68g")
OPENAI_MODEL = os.environ.get("KP_OPENAI_MODEL", "glm-5")
MAX_AGENT_TURNS = int(os.environ.get("KP_MAX_AGENT_TURNS", "50"))
