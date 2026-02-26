"""I/O helpers for experiments and JSON/pickle serialization."""

import json
import os
import pickle
from typing import Any, Dict

import numpy as np

def save_experiment(
    data: Dict[str, Any],
    filename: str,
    compress: bool = False,
) -> str:
    """
    Save a complete experiment payload.
    
    Args:
        data: Dictionary containing results
        filename: Output path
        compress: If True, use pickle (more compact)
    
    Returns:
        Absolute path of the saved file
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    if compress:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    else:
        # JSON with NumPy conversion
        data_serializable = _make_json_serializable(data)
        with open(filename, 'w') as f:
            json.dump(data_serializable, f, indent=2)
    
    abs_path = os.path.abspath(filename)
    print(f"✓ Experiment saved : {abs_path}")
    
    return abs_path


def load_experiment(filename: str) -> Dict[str, Any]:
    """
    Load an experiment payload.
    
    Args:
        filename: File path
    
    Returns:
        Dictionary with data
    """
    if filename.endswith('.pkl'):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    else:
        with open(filename, 'r') as f:
            data = json.load(f)
        # Convert lists back to arrays
        data = _make_numpy_arrays(data)
    
    print(f"✓ Experiment loaded : {filename}")
    
    return data


def _make_json_serializable(obj: Any) -> Any:
    """Convert NumPy arrays to JSON-serializable lists."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    else:
        return obj


def _make_numpy_arrays(obj: Any) -> Any:
    """Convert nested Python lists back to NumPy arrays when possible."""
    if isinstance(obj, dict):
        return {k: _make_numpy_arrays(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        try:
            return np.array(obj)
        except (ValueError, TypeError):
            return [_make_numpy_arrays(v) for v in obj]
    else:
        return obj
