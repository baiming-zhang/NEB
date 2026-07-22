from pathlib import Path
import sys

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid_sss.basisnet.tg3d.training import main  # noqa: E402


if __name__ == "__main__":
    main()
