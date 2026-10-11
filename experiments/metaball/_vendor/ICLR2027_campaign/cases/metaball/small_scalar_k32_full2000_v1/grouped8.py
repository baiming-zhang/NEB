
from original_potential_k32_core import cp,Jet1,curl1,curl_adjoint

def network_fields(model,data,which='aud',chunk=2048):
    result={k:cp.empty(((data.n if k=='d' else 3*data.n),model.K),dtype='f8') for k in which}
    for first in range(0,data.n,chunk):
        stop=min(first+chunk,data.n);p=data.part(first,stop)
        for mode in range(0,model.K,8):
            f,_=model.forward(p,mode,mode+8)
            if 'a' in result:result['a'][3*first:3*stop,mode:mode+8]=f.value.reshape(8,-1).T
            if 'u' in result:result['u'][3*first:3*stop,mode:mode+8]=curl1(f).reshape(8,-1).T
            if 'd' in result:result['d'][first:stop,mode:mode+8]=sum(f.grad[...,j,j] for j in range(3)).T
    return result

def network_backward(model,data,value=None,curl=None,div=None,grads=None,chunk=2048):
    if grads is None:grads=([cp.zeros(w.shape,dtype='f8') for w in model.W],[cp.zeros(b.shape,dtype='f8') for b in model.b])
    gw,gb=grads
    for first in range(0,data.n,chunk):
        stop=min(first+chunk,data.n);p=data.part(first,stop);n=stop-first
        for mode in range(0,model.K,8):
            _,cache=model.forward(p,mode,mode+8,True)
            av=cp.zeros((8,n,3),dtype=data.x.dtype);ag=cp.zeros((8,n,3,3),dtype=data.x.dtype)
            if value is not None:av[:]=value[3*first:3*stop,mode:mode+8].T.reshape(8,n,3)
            if curl is not None:ag+=curl_adjoint(curl[3*first:3*stop,mode:mode+8].T.reshape(8,n,3).astype(data.x.dtype))
            if div is not None:
                for j in range(3):ag[...,j,j]+=div[first:stop,mode:mode+8].T
            dw,db=model.backward(p,mode,mode+8,cache,Jet1(av,ag))
            for j in range(len(gw)):gw[j][mode:mode+8]+=dw[j];gb[j][mode:mode+8]+=db[j]
    return grads
