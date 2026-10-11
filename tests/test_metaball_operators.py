"""Full-capacity mass, viscous and packed-quadratic CPU oracle."""
import numpy as np
from experiments.metaball import operators as o


def test_full_M_A_Q_and_packed_rhs(tmp_path, monkeypatch):
    # Resource thresholds belong to production runs; this numerical unit test
    # uses 137 points and all five modes on any CI memory configuration.
    monkeypatch.setattr(o, "guard", lambda: None)
    rng=np.random.default_rng(17);N=137;K=5;n=8
    r=tmp_path;flat=np.sort(rng.choice(n**3,N,replace=False));B=rng.normal(size=(N,3,K));normal=rng.normal(size=(n,n,n,3));normal/=np.linalg.norm(normal,axis=3,keepdims=True);chi=rng.uniform(size=(n,n,n))
    for name,value in [('B',B),('flat',flat),('normal',normal),('chi',chi)]:np.save(r/(name+'.npy'),value)
    derived=o.derive(r/'B.npy',r/'flat.npy',r/'normal.npy',r/'chi.npy',r/'derived',grid=n)
    d=np.load(derived);sp=o.Spectral(normal,chi,n)
    for mode in range(K):
        u=np.zeros((n**3,3));u[flat]=B[:,:,mode];u=u.reshape(n,n,n,3).transpose(3,0,1,2);v=sp.fft(u)
        direct=[sp.real(sp.project(v)),sp.real(sp.curl(v))];components=sp.momentum(u,True);direct.append(components['viscous']+components['normal_penalty'])
        expect=np.concatenate([x.transpose(1,2,3,0).reshape(-1,3)[flat] for x in direct],axis=1)
        np.testing.assert_allclose(d[:,:,mode],expect,rtol=1e-14,atol=1e-13)
    operators,Q=o.assemble(r/'B.npy',derived,r/'assembly',grid=n,point_block=31,slab_bytes=8*K*K*2)
    z=np.load(operators);q=np.load(Q);weight=1/n**3
    np.testing.assert_allclose(z['M'],weight*np.einsum('ndi,ndj->ij',B,B),rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(z['A'],weight*np.einsum('ndi,ndj->ij',B,d[:,6:9]),rtol=1e-13,atol=1e-13)
    raw=np.empty((K,K,K))
    for j in range(K):
        for k in range(K):raw[j,:,k]=np.einsum('ndi,nd->i',d[:,:3],np.cross(B[:,:,j],d[:,3:6,k]))*weight
    expected=np.column_stack([raw[j,:,k] if j==k else raw[j,:,k]+raw[k,:,j] for j,k in zip(*np.triu_indices(K))])
    np.testing.assert_allclose(q,expected,rtol=1e-12,atol=1e-12)
    c=rng.normal(size=K);np.testing.assert_allclose(q@(c[np.triu_indices(K)[0]]*c[np.triu_indices(K)[1]]),np.einsum('jik,j,k->i',raw,c,c),rtol=1e-12,atol=1e-12)
    z.close()
    try:o.assemble(r/'B.npy',derived,r/'assembly',grid=n)
    except FileExistsError:pass
    else:raise AssertionError('Existing run overwritten')
