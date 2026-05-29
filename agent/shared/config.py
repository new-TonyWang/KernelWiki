"""Agent configuration — paths adapted for the merged KernelWiki layout."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KNOWLEDGE_ROOT = PROJECT_ROOT  # KernelWiki root IS the knowledge root now
SOURCE_CORPUS_ROOT = PROJECT_ROOT / "corpus"
WIKI_ROOT = PROJECT_ROOT / "wiki" / "nvidia"
SOURCES_ROOT = PROJECT_ROOT / "sources"
REASONING_ROOT = PROJECT_ROOT / "reasoning"
TEMPLATES_ROOT = PROJECT_ROOT / "templates"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
