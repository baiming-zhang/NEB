"""Disjoint16384-point held-out geometry reconstruction, full256 trial space."""
from pathlib import Path
import argparse,json,numpy as np
K=256
def write(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')
class SavedBasis:
    def __init__(self,fit,assessment):self.fit=fit;self.assessment=assessment
    def matrix(self,points,geometry):
        ids=points[:,0].astype('i8');which=int(points[0,1]);b=self.fit if which==0 else self.assessment
        return cp.asarray(b[ids],dtype='f8').reshape(-1,K)

def reconstruct(basis, samples, target):
    target.mkdir(parents=True, exist_ok=True)
    M = cp.zeros((K, K), dtype='f8')
    rhs = cp.zeros(K, dtype='f8')
    norm = 0.0
    for off in range(0, 16384, 1024):
        U = basis.matrix(samples['fit_points'][off:off + 1024], samples['geometry']).astype('f8')
        y = cp.asarray(samples['fit_target'][off:off + 1024].reshape(-1), dtype='f8')
        M += U.T @ U
        rhs += U.T @ y
        norm += float(y @ y)
    s = cp.sqrt(cp.diag(M))
    H = M / s[:, None] / s[None, :]
    ev = cp.linalg.eigvalsh((H + H.T) * 0.5)
    assert float(ev[0]) > 1e-10, 'AllK rank failure; no truncation'
    L = cp.linalg.cholesky((H + H.T) * 0.5)
    a = solve_triangular(L.T, solve_triangular(L, rhs / s, lower=True), lower=False) / s
    res = float(cp.linalg.norm(M @ a - rhs) / cp.maximum(cp.linalg.norm(rhs), 1e-300))
    assert res < 1e-08
    predicted = []
    for off in range(0, 16384, 1024):
        U = basis.matrix(samples['assessment_points'][off:off + 1024], samples['geometry']).astype('f8')
        predicted.append(cp.asnumpy(U @ a).reshape(-1, 3))
    prediction = np.concatenate(predicted)
    truth = samples['assessment_target'].astype('f8')
    metrics = dict(K=K, fit_points=16384, assessment_points=16384, assessment_relative_L2=float(np.linalg.norm(prediction - truth) / np.linalg.norm(truth)), fit_relative_L2=float(np.sqrt(max(0, float(norm - 2 * a @ rhs + a @ M @ a)) / norm)), normal_residual=res, scaled_condition=float(ev[-1] / ev[0]), full_mass=True, no_ridge_no_truncation=True)
    np.savez(target / 'arrays.npz', M=cp.asnumpy(M), rhs=cp.asnumpy(rhs), coefficients=cp.asnumpy(a), prediction=prediction)
    write(target / 'metrics.json', metrics)
    return metrics

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--fit-basis',required=True);p.add_argument('--assessment-basis',required=True);p.add_argument('--samples',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();global cp,solve_triangular
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular
    fit=np.load(a.fit_basis,mmap_mode='r',allow_pickle=False);assessment=np.load(a.assessment_basis,mmap_mode='r',allow_pickle=False)
    if fit.shape!=assessment.shape or fit.shape!=(16384,3,256) or not np.isfinite(fit).all() or not np.isfinite(assessment).all():raise ValueError('Full256 fields on both disjoint16384 samples required')
    with np.load(a.samples,allow_pickle=False) as z:samples={k:z[k] for k in z.files}
    samples['fit_points']=np.column_stack((np.arange(16384),np.zeros(16384)));samples['assessment_points']=np.column_stack((np.arange(16384),np.ones(16384)))
    reconstruct(SavedBasis(fit,assessment),samples,Path(a.output))
if __name__=='__main__':main()
