"""Configuration file helpers."""

import json
import os
from typing import Any, Dict

def save_config(config: Dict[str, Any], filename: str) -> None:
    """Save a configuration dictionary."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Configuration saved : {filename}")


def load_config(filename: str) -> Dict[str, Any]:
    """Load a configuration dictionary."""
    with open(filename, 'r') as f:
        config = json.load(f)
    print(f"✓ Configuration loaded : {filename}")
    return config


def merge_configs(default: Dict[str, Any], custom: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configs (custom overrides default)."""
    result = default.copy()
    result.update(custom)
    return result
