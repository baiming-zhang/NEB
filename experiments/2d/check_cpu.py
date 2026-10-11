"""Small CPU mathematics checks. Never imports a CuPy backend."""
from pathlib import Path
import importlib.util,json,sys
import numpy as np
BASE=Path(__file__).resolve().parent/'_vendor'
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def main():
    dns=load(BASE/'native_periodic/2D_Vortex_Merger/scripts/generate_gt.py','neb_cpu_dns')
    n=16;solver=dns.VorticityDNS(n,2*np.pi,.0010742913669168762,.005)
    x,y=np.meshgrid(np.arange(n)*2*np.pi/n,np.arange(n)*2*np.pi/n)
    omega=np.sin(2*x)*np.cos(y);hat=solver.filtered_hat(omega)
    w,psi,u,v=solver.fields_from_hat(hat)
    divergence=np.fft.ifft2(1j*solver.kx*np.fft.fft2(u)+1j*solver.ky*np.fft.fft2(v)).real
    curl=np.fft.ifft2(1j*solver.kx*np.fft.fft2(v)-1j*solver.ky*np.fft.fft2(u)).real
    assert np.max(abs(divergence))<1e-12
    # Original solver's scalar vorticity convention and velocity are tested directly.
    assert np.max(abs(curl-w))<1e-12
    assert np.isfinite(solver.step(hat)).all()
    mod=load(BASE/'ICLR2027_campaign/cases/topology3/random_streamfunction_20260928/preview_random_streamfunction.py','neb_cpu_quintic')
    field=mod.RandomField('one_circle',0);points=np.array([[2.3,2.2],[7.2,-2.1],[3.1,-1.9]])
    a=field.evaluate(points);eps=1e-5
    dx=(field.evaluate(points+[eps,0])[:,0]-field.evaluate(points-[eps,0])[:,0])/(2*eps)
    dy=(field.evaluate(points+[0,eps])[:,0]-field.evaluate(points-[0,eps])[:,0])/(2*eps)
    assert np.allclose(a[:,1],dy,rtol=2e-6,atol=1e-8)
    assert np.allclose(a[:,2],-dx,rtol=2e-6,atol=1e-8)
    rng=np.random.default_rng(77);B=rng.normal(size=(137,8));T=rng.normal(size=(137,20));w=rng.uniform(.1,1,137)
    M=np.zeros((8,8));r=np.zeros((8,20))
    for first in range(0,137,31):
        b=B[first:first+31];t=T[first:first+31];v=w[first:first+31]
        M+=b.T@(v[:,None]*b);r+=b.T@(v[:,None]*t)
    c=np.linalg.solve(M,r);d=np.sum(w[:,None]*(B@c-T)**2,axis=0)
    assert np.allclose(d,np.sum(w[:,None]*T*T,axis=0)-np.sum(c*r,axis=0),rtol=1e-13)
    core=load(Path(__file__).with_name('nkb')/'core.py','neb_cpu_nkb')
    model=core.Model(np,4,2,2,seed=3,width=8,hidden=7,dtype='float64')
    q=np.array([[.3,.4],[1.2,2.3]]);f,j=core.features(q,np,True)
    values,jac,tape=model.forward(f,j,True)
    e=1e-6;fp,jp=core.features(q+[e,0],np,True);fm,jm=core.features(q-[e,0],np,True)
    vp=model.forward(fp,jp)[0];vm=model.forward(fm,jm)[0]
    assert np.allclose((vp-vm)/(2*e),jac[:,:,:,0],rtol=1e-5,atol=1e-8)
    raw,gv,gj=core.losses(values,jac,np.ones(2),np.zeros(2),np.zeros_like(q),np)
    grads=model.backward(tape,gv,gj);model.adam(grads,.0005);assert model.step==1
    assert len(model.W)+len(model.b)+len(model.first)+len(model.second)==48
    assert 'cupy' not in sys.modules
    print(json.dumps(dict(passed=True,periodic_analytic_divergence=True,quintic_analytic_derivative=True,full_mass_direct20=True,NKB_analytic_Jacobian_and_full48_Adam=True,CUDA_imported=False,tiny_interface_only=True)))
if __name__=='__main__':main()
