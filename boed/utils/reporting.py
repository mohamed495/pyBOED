"""Console reporting helpers for metrics and comparisons."""

from typing import Any, Dict

import numpy as np

def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """
    Print formatted metrics.
    
    Args:
        metrics: Dict {name: value}
        title: Report title
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    for name, value in metrics.items():
        if isinstance(value, float):
            if abs(value) < 1e-6 or abs(value) > 1e6:
                print(f"  {name:<30} {value:>15.6e}")
            else:
                print(f"  {name:<30} {value:>15.6f}")
        else:
            print(f"  {name:<30} {value:>15}")
    
    print(f"{'='*60}\n")


def comparison_table(
    results: list[Dict[str, Any]],
    columns: list[str],
    title: str = "Comparison",
) -> str:
    """
    Generate an ASCII comparison table.
    
    Args:
        results: List of result dictionaries
        columns: Columns to display
        title: Title
    
    Returns:
        Formatted string (ASCII table)
    """
    # Determine column widths
    widths = {col: len(col) for col in columns}
    for result in results:
        for col in columns:
            val_str = str(result.get(col, ""))
            widths[col] = max(widths[col], len(val_str))
    
    # Header
    header = " | ".join(f"{col:<{widths[col]}}" for col in columns)
    separator = "-" * (len(header) + (len(columns) - 1) * 3)
    
    output = f"\n{title}\n{separator}\n{header}\n{separator}\n"
    
    # Rows
    for result in results:
        row = " | ".join(
            f"{str(result.get(col, '')):<{widths[col]}}"
            for col in columns
        )
        output += row + "\n"
    
    output += separator + "\n"
    
    return output
