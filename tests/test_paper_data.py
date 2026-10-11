from pathlib import Path
import json
import numpy as np
from neb.figures import paired_statistics

ROOT = Path(__file__).resolve().parents[1]


def test_paired_fields_and_sample_statistics():
    stats = paired_statistics(ROOT / "data/paper/metaball20.csv")
    assert set(stats) == {(method, k) for method in ["LEB", "VRQ", "CCQ", "NKB"]
                          for k in [32, 64, 128, 256]}
    # Selected published means independently pin the paired-field data.
    np.testing.assert_allclose(stats["CCQ", 32][0], 0.8426269234109827, rtol=0, atol=2e-15)
    np.testing.assert_allclose(stats["CCQ", 64][1], 0.0010720684309286383,
                               rtol=0, atol=2e-15)


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
