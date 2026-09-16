"""Independently FFT-differentiate exact Fourier potential/velocity modes."""
import numpy as np
from verify_analytic import ROOT,waves,write_json

OUT=ROOT/'outputs/comparison_64_1024'
axis=np.arange(128)*2*np.pi/128
x,y=np.meshgrid(axis,axis,indexing='xy')
f=np.fft.fftfreq(128)*128; fx,fy=np.meshgrid(f,f); f2=fx*fx+fy*fy
def lap(a):
    return np.fft.ifft2(f2[...,None]*np.fft.fft2(a,axes=(0,1)),axes=(0,1)).real

results={}
for k in (64,1024):
    pairs=waves(k); rp=[]; ru=[]; eigenvalues=[]
    for start in range(0,len(pairs),32):
        p=pairs[start:start+32]
        phase=x[...,None]*p[:,0]+y[...,None]*p[:,1]
        phi=np.stack((np.cos(phase),np.sin(phase)),axis=-1).reshape(128,128,-1)*np.sqrt(2)/(2*np.pi)
        d=np.stack((-np.sin(phase),np.cos(phase)),axis=-1).reshape(128,128,-1)*np.sqrt(2)/(2*np.pi)
        kx=np.repeat(p[:,0],2); ky=np.repeat(p[:,1],2); ev=kx*kx+ky*ky
        u=d*ky/np.sqrt(ev); v=-d*kx/np.sqrt(ev)
        potential=np.sqrt(np.sum((lap(phi)-phi*ev)**2,axis=(0,1)))/(ev*np.sqrt(np.sum(phi*phi,axis=(0,1))))
        velocity=np.sqrt(np.sum((lap(u)-u*ev)**2+(lap(v)-v*ev)**2,axis=(0,1)))/(ev*np.sqrt(np.sum(u*u+v*v,axis=(0,1))))
        rp.extend(potential.tolist()); ru.extend(velocity.tolist()); eigenvalues.extend(ev.tolist())
    assert max(rp+ru)<1e-9
    results[str(k)]=dict(modes=k,eigenvalues=eigenvalues,eigenvalue_relative_error_mean=0.,
        potential_relation_residual_mean=float(np.mean(rp)),potential_relation_residual_max=max(rp),
        velocity_relation_residual_mean=float(np.mean(ru)),velocity_relation_residual_max=max(ru),
        potential_relation_residual=rp,velocity_relation_residual=ru,
        verification='Exact Fourier eigenvalues m^2+n^2; independently evaluated FFT Laplacian of analytic fields')
    np.savetxt(OUT/f'laplacian_k{k}_eigenrelation.csv',np.column_stack((np.arange(1,k+1),eigenvalues,rp,ru)),delimiter=',',
               header='rank,eigenvalue,potential_residual,velocity_residual',comments='')
write_json(OUT/'analytic_eigenrelation.json',results)
print({key:{name:val for name,val in value.items() if 'mean' in name or 'max' in name} for key,value in results.items()})
