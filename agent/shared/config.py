"""Agent configuration — paths and runtime constants for the merged KernelWiki layout.

All paths point to the new KernelWiki directory structure.
Runtime constants are environment-overridable via KP_* (preferred) or legacy names.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KNOWLEDGE_ROOT = PROJECT_ROOT  # KernelWiki root IS the knowledge root
SOURCE_CORPUS_ROOT = PROJECT_ROOT / "corpus"
SOURCE_CORPUS_INDEX_ROOT = SOURCE_CORPUS_ROOT / "INDEX"
SOURCE_CORPUS_MANIFEST = SOURCE_CORPUS_ROOT / "MANIFEST.yaml"
WIKI_ROOT = PROJECT_ROOT / "wiki" / "nvidia"
SOURCES_ROOT = PROJECT_ROOT / "sources"
REASONING_ROOT = PROJECT_ROOT / "reasoning"
TEMPLATES_ROOT = PROJECT_ROOT / "templates"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

# Remote GPU execution settings — KP_* takes precedence, KB_* as legacy fallback
REMOTE_HOST = os.environ.get("KP_REMOTE_HOST",
              os.environ.get("KB_REMOTE_HOST", "localhost"))
REMOTE_DIR = os.environ.get("KP_REMOTE_DIR",
             os.environ.get("KB_REMOTE_DIR", "/tmp/kb-agent"))
REMOTE_CUDA_PATH = os.environ.get("KP_REMOTE_CUDA_PATH",
                   os.environ.get("KB_REMOTE_CUDA_PATH", "/usr/local/cuda/bin"))
REMOTE_PYLIB = os.environ.get("KP_REMOTE_PYLIB",
               os.environ.get("KB_REMOTE_PYLIB", "python3"))

# OpenAI path settings — KP_* takes precedence per agent/openai_path/README.md
MAX_AGENT_TURNS = int(os.environ.get("KP_MAX_AGENT_TURNS",
                      os.environ.get("KB_MAX_AGENT_TURNS", "50")))
OPENAI_API_KEY = os.environ.get("KP_OPENAI_API_KEY",
                 os.environ.get("OPENAI_API_KEY", "sk-dummy"))
OPENAI_BASE_URL = os.environ.get("KP_OPENAI_BASE_URL",
                  os.environ.get("OPENAI_BASE_URL", "http://localhost:4000/v1"))
OPENAI_MODEL = os.environ.get("KP_OPENAI_MODEL",
               os.environ.get("OPENAI_MODEL", "gpt-4o"))
