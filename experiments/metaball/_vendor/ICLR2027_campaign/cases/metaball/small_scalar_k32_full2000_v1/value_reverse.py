
import cupy as cp

def forward_value(model,data,start,stop):
    x=data.features.value;history=[]
    for j in range(len(model.W)-1):
        omega=model.c.siren_first_omega0 if j==0 else model.c.siren_hidden_omega0
        pre=(cp.matmul(x,model.W[j][start:stop].transpose(0,2,1))+model.b[j][start:stop,None,:])*omega
        history.append((x,pre,omega));x=cp.sin(pre)
    raw=cp.matmul(x,model.W[-1][start:stop].transpose(0,2,1))+model.b[-1][start:stop,None,:]
    return data.envelope.value*raw,(x,history)

def linear_reverse(source,a,W):
    expr='bno,nf->bof' if source.ndim==2 else 'bno,bnf->bof'
    dw=cp.einsum(expr,a,source,dtype=cp.float32)
    db=cp.sum(a,axis=1,dtype=cp.float32)
    return cp.matmul(a,W,dtype=cp.float32),dw,db

def backward_value(model,data,start,stop,cache,value):
    source,history=cache;a=value*data.envelope.value
    a,dw,db=linear_reverse(source,a,model.W[-1][start:stop]);weights=[dw];biases=[db]
    for j in range(len(history)-1,-1,-1):
        source,pre,omega=history[j];a=a*cp.cos(pre);a=a*omega
        a,dw,db=linear_reverse(source,a,model.W[j][start:stop]);weights.append(dw);biases.append(db)
    return weights[::-1],biases[::-1]

def network_backward_value(model,data,value,*,grads=None):
    if model.K!=32 or data.x.dtype!=cp.float32 or value.shape!=(3*data.n,32):raise ValueError('Requires original K32 FP32 data and complete value cotangent')
    if grads is None:grads=([cp.zeros(w.shape,dtype='f8') for w in model.W],[cp.zeros(b.shape,dtype='f8') for b in model.b])
    gw,gb=grads
    for first in range(0,data.n,2048):
        stop=min(first+2048,data.n);p=data.part(first,stop);n=stop-first
        for mode in range(0,32,8):
            _,cache=forward_value(model,p,mode,mode+8)
            av=cp.zeros((8,n,3),dtype=data.x.dtype);av[:]=value[3*first:3*stop,mode:mode+8].T.reshape(8,n,3)
            dw,db=backward_value(model,p,mode,mode+8,cache,av)
            for j in range(len(gw)):gw[j][mode:mode+8]+=dw[j];gb[j][mode:mode+8]+=db[j]
    return grads
