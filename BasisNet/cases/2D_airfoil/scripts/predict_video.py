"""Canonical continuous GT/reconstruction video entry point.

This wrapper keeps the public case layout parallel to ``2D_karman`` while
the implementation remains in ``run_continuous_postprocess.py``.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("run_continuous_postprocess.py")),
        run_name="__main__",
    )
