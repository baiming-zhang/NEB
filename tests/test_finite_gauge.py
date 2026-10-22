"""CPU checks of the finite-gauge quotient and its final-column cotangent."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


class CPUArrayAPI:
    cuda=SimpleNamespace(Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda:None)))
    def __getattr__(self,name):return getattr(np,name)
    @staticmethod
    def asnumpy(a):return np.asarray(a)
    @staticmethod
    def get_default_memory_pool():return SimpleNamespace(free_all_blocks=lambda:None)


def test_finite_gauge_rayleigh_coordinates_and_reverse():
    spec=importlib.util.spec_from_file_location('finite_gauge',ROOT/'experiments/metaball/finite_gauge.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    objective=module.original_objective(CPUArrayAPI())
    rng=np.random.default_rng(6731);n=17
    a=rng.normal(size=(3*n,4));u=rng.normal(size=(3*n,4))
    result=objective(a,u,n)
    np.testing.assert_allclose(a@result['T'],result['Q'],atol=2e-15)
    np.testing.assert_allclose(u@result['T'],result['U'],atol=2e-15)
    np.testing.assert_allclose(result['Q'].T@result['Q']/n,np.eye(4),atol=5e-15)
    np.testing.assert_allclose(result['losses'],np.sum(result['U']**2,axis=0)/n,atol=5e-15)
    last=result['order'][-1];da=np.zeros_like(a);du=np.zeros_like(u)
    da[:,last]=rng.normal(size=3*n);du[:,last]=rng.normal(size=3*n)
    eps=1e-6
    finite=(objective(a+eps*da,u+eps*du,n)['losses'].sum()-objective(a-eps*da,u-eps*du,n)['losses'].sum())/(2*eps)
    analytic=np.sum(result['GA']*da+result['GU']*du)
    np.testing.assert_allclose(analytic,finite,rtol=2e-7,atol=2e-9)
