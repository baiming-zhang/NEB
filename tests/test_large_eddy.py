"""The paper's independent Fourier potential and physical curl prescription."""
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def test_philox_keys_and_analytic_vector_potential_curl():
    spec=importlib.util.spec_from_file_location('large_eddy',ROOT/'experiments/metaball/large_eddy.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    p=json.loads((ROOT/'experiments/metaball/configs/large_eddy20.json').read_text())
    keys=[int(k) for k in p['rng_keys']]
    assert keys==list(range(122093513105536090505257636610083695833,122093513105536090505257636610083695853))
    assert np.array_equal(m.WAVEVECTORS,p['wavevectors']) and m.WAVEVECTORS.shape==(16,3)
    raw=m.draw(keys[0]);assert raw.shape==(16,2,3)
    np.testing.assert_array_equal(raw,m.draw(keys[0]))
    assert not np.array_equal(raw,m.draw(keys[1]))
    x=np.random.default_rng(86).random((9,3))
    def potential(q):
        phase=2*np.pi*q@m.WAVEVECTORS.T
        return np.cos(phase)@raw[:,0,:]+np.sin(phase)@raw[:,1,:]
    h=2e-6
    d=[(potential(x+np.eye(3)[a]*h)-potential(x-np.eye(3)[a]*h))/(2*h) for a in range(3)]
    curl=np.stack((d[1][:,2]-d[2][:,1],d[2][:,0]-d[0][:,2],d[0][:,1]-d[1][:,0]),axis=1)
    np.testing.assert_allclose(m.curl_points(raw,x),curl,rtol=2e-8,atol=3e-8)
