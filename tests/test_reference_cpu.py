from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_public_fem_and_sparse_controls():
    pytest.importorskip("meshpy")
    for module in ["experiments.multihole.reference.check_cpu",
                   "experiments.multihole.reference.check_sparse_cpu"]:
        subprocess.run([sys.executable, "-B", "-m", module], cwd=ROOT,
                       check=True, capture_output=True, text=True)
