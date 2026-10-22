from pathlib import Path
import json
import numpy as np
from neb.figures import paired_statistics

ROOT = Path(__file__).resolve().parents[1]


def test_paired_fields_and_sample_statistics():
    stats = paired_statistics(ROOT / "data/paper/metaball20.csv")
    assert set(stats) == {(method, k) for method in ["Laplacian", "Velocity", "Potential", "NKB"]
                          for k in [32, 64, 128, 256, 512]} | {
                              (method, 1024) for method in ["Laplacian", "Velocity", "NKB"]}
    # Selected published means independently pin the paired-field data.
    np.testing.assert_allclose(stats["Potential", 512][0], 0.2373323, rtol=0, atol=5e-8)
    np.testing.assert_allclose(stats["Laplacian", 256][0], 0.2651102, rtol=0, atol=5e-8)


def test_periodic_assessment_configuration():
    data = json.loads((ROOT / "data/paper/periodic_merger.json").read_text(encoding="utf-8"))
    protocol = data["protocol"]
    assert protocol["K"] == 1024 and protocol["frames"] == 241
    assert protocol["quadrature_N"] == 256 and protocol["observation_N"] == 128
    np.testing.assert_allclose(protocol["final_time"], 18 * np.pi)


def test_data_has_no_machine_locations():
    import re
    for path in (ROOT / "data/paper").glob("*.json"):
        assert not re.search(r"[A-Za-z]:[\\/]", path.read_text(encoding="utf-8")), path
