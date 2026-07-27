from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eigenfluid_sss.basisnet.parametric_wake3d.training import main  # noqa: E402


if __name__ == "__main__":
    main()
