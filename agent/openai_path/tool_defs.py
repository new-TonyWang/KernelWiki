"""OpenAI function-calling tool definitions for the KB-gen agent."""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "source_search",
            "description": (
                "Search the manifest-backed 05-source-corpus and return structured JSON hits. "
                "Use this for official docs, blogs, whitepapers, and source repos instead of "
                "grepping absolute external paths directly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term or regex pattern.",
                    },
                    "scope": {
                        "type": "string",
                        "description": "Logical source scope like 'cuda-official', 'blogs/colfax', or 'source-code/cutlass'.",
                    },
                    "source_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional filter, e.g. ['cuda-official'] or ['source-code'].",
                    },
                    "entity_kind": {
                        "type": "string",
                        "description": "Optional intent hint: api | feature | operator | general.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of hits to return.",
                    },
                    "regex": {
                        "type": "boolean",
                        "description": "Interpret query as a regex when true.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "source_read",
            "description": (
                "Read a source corpus file by line anchor or section heading and return structured JSON. "
                "Use after source_search before making code-facing decisions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "A source corpus path returned by source_search or a known 05-source-corpus path.",
                    },
                    "anchor": {
                        "type": "string",
                        "description": "Optional anchor such as L120-L140 or #section-heading.",
                    },
                    "section": {
                        "type": "string",
                        "description": "Optional section title / heading text.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return.",
                    },
                    "focused_query": {
                        "type": "string",
                        "description": "Optional focused term to trim the returned section near the most relevant passage.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "source_list",
            "description": "List source corpus entries available in MANIFEST.yaml.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "description": "Optional logical scope prefix.",
                    },
                    "source_type": {
                        "type": "string",
                        "description": "Optional source type filter.",
                    },
                    "keyword": {
                        "type": "string",
                        "description": "Optional keyword filter over title, aliases, and tags.",
                    },
                    "year": {
                        "type": "string",
                        "description": "Reserved optional year filter.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of entries to return.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "source_resolve",
            "description": "Resolve a source id or 05-source-corpus path to manifest metadata / actual local path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "Source id, alias, logical root, or 05-source-corpus path.",
                    },
                },
                "required": ["ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "provenance_walk",
            "description": "Read frontmatter source refs from a knowledge entry and return them as JSON.",
            "parameters": {
                "type": "object",
                "properties": {
                    "knowledge_path": {
                        "type": "string",
                        "description": "Knowledge file path under this repo.",
                    },
                },
                "required": ["knowledge_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "provenance_back",
            "description": "Find which knowledge entries reference a given source corpus path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path under 05-source-corpus as stored in frontmatter source.path.",
                    },
                    "anchor": {
                        "type": "string",
                        "description": "Optional anchor filter.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_source",
            "description": (
                "Search for a regex pattern in files under a directory using ripgrep. "
                "Returns matching lines with file:line: prefix. Use for upstream doc search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern (ripgrep syntax). Escape special chars with backslash.",
                    },
                    "scope": {
                        "type": "string",
                        "description": "Absolute directory path to search under (must be in task upstream_scope).",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Optional file glob filter, e.g. '*.md', '*.cu', '*.cuh'.",
                    },
                },
                "required": ["pattern", "scope"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file and return its text content (truncated at 500 lines).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path (absolute or relative to project root).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write content to a file. Creates parent directories if needed. "
                "Only paths under the project root are allowed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path (absolute or relative to project root).",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and subdirectories in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "Run a shell command locally on the dev machine. "
                "Use for lint, probe_gap_scan, file operations. "
                "NOT for CUDA/nvcc — use run_on_gpu for that."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_on_gpu",
            "description": (
                "Run a command on the remote H200 GPU host via SSH. "
                "PATH and PYTHONPATH are auto-configured for CUDA 12.9 + pylib. "
                "Use for: nvcc compilation, running CUDA binaries, kp_introspect. "
                "Always sync_to_gpu first if you changed local files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to run in the remote kernel-kb-mvp directory.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sync_to_gpu",
            "description": "Rsync local project files to the remote GPU host. Call before run_on_gpu if local files changed.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_provenance",
            "description": "Rebuild 05-source-corpus/INDEX/provenance-back.jsonl from frontmatter source refs.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sync_from_gpu",
            "description": "Rsync files from the remote GPU host back to local. Call after run_on_gpu produces output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "remote_subpath": {
                        "type": "string",
                        "description": "Subpath to sync (default: 'knowledge/'). E.g. 'knowledge/80-experience/'.",
                    },
                },
            },
        },
    },
]
