"""Pytest bootstrap to guarantee imports come from the local workspace."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _is_under_root(path: str | None) -> bool:
    if not path:
        return False
    try:
        return str(Path(path).resolve()).startswith(str(ROOT))
    except OSError:
        return False


def pytest_sessionstart(session) -> None:  # noqa: D401
    # Ensure local repository root is first on sys.path.
    root_str = str(ROOT)
    if not sys.path or sys.path[0] != root_str:
        if root_str in sys.path:
            sys.path.remove(root_str)
        sys.path.insert(0, root_str)

    # If boed was preloaded from site-packages, purge it so local package wins.
    boed_mod = sys.modules.get("boed")
    if boed_mod is not None and not _is_under_root(getattr(boed_mod, "__file__", None)):
        for name in list(sys.modules):
            if name == "boed" or name.startswith("boed."):
                del sys.modules[name]
