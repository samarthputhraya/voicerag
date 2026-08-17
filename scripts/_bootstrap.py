"""Make ``voicerag`` and ``eval`` importable when a script is run directly.

The package is deliberately not installed into the environment -- ``pytest``
puts ``src`` on the path via ``pyproject.toml``, and a judge cloning the repo
should be able to run ``python scripts/smoke.py`` without a ``pip install -e``
step first. This module reproduces exactly that path setup for the CLIs, and
nothing else.

Import it before any ``voicerag``/``eval`` import in a script::

    from _bootstrap import bootstrap

    ROOT = bootstrap()
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["bootstrap"]


def bootstrap() -> Path:
    """Prepend the repository root and ``src`` to ``sys.path``.

    Returns:
        The repository root, so callers can resolve default output paths
        relative to the checkout rather than to the current directory.
    """
    root = Path(__file__).resolve().parents[1]
    for entry in (root / "src", root):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)
    return root
