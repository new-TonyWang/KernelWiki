"""task_schema — load and validate task YAML files (N3 contract)."""

from __future__ import annotations

from pathlib import Path

import yaml

REQUIRED_FIELDS = {
    "task_id", "task_type", "target_path", "hardware",
    "upstream_scope", "references", "success_criteria",
}
VALID_TASK_TYPES = {"build-skill", "probe-api", "build-pattern", "introspect-hardware"}


def load_task(path: Path) -> dict:
    """Load a task YAML, validate required fields, return the dict."""
    with open(path) as f:
        task = yaml.safe_load(f)

    if not isinstance(task, dict):
        raise ValueError(f"Task file {path} is not a YAML mapping")

    missing = REQUIRED_FIELDS - set(task.keys())
    if missing:
        raise ValueError(f"Task {path} missing required fields: {sorted(missing)}")

    if task["task_type"] not in VALID_TASK_TYPES:
        raise ValueError(
            f"Invalid task_type '{task['task_type']}'; "
            f"must be one of {sorted(VALID_TASK_TYPES)}"
        )

    if not isinstance(task["upstream_scope"], list):
        raise ValueError("upstream_scope must be a list of paths")

    return task


def task_to_yaml_str(task: dict) -> str:
    """Dump a task dict back to a clean YAML string for embedding in prompts."""
    return yaml.dump(task, default_flow_style=False, sort_keys=False, allow_unicode=True)
