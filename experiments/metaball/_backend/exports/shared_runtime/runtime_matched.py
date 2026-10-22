






def install(cm,variant='minimal'):
    if variant not in ('literal','minimal'):raise ValueError(variant)
    grouped=cm.rt.grouped8;cls=cm.train.Model
    original_fields=grouped.network_fields;original_backward=grouped.network_backward
    ns=cls.forward.__globals__;cp=cm.cp;Jet1=ns['Jet1'];old=ns['old']
    linear1=ns['linear1'];scale1=ns['scale1'];curl1=ns['curl1'];curl_adjoint=ns['curl_adjoint']
    def pack(j):
        axis=1 if j.value.ndim==3 else 0
        return cp.concatenate([j.value,*[j.grad[...,i] for i in range(3)]],axis=axis)
    def linear_reverse(source,a,W,discard_input=False):
        ap=pack(a);xp=pack(source)
        dw=ap.transpose(0,2,1)@xp;db=a.value.sum(axis=1)
        if discard_input:return None,dw,db
        z=ap@W;n=source.value.shape[-2];z=z.reshape(len(W),4,n,W.shape[-1])
        return Jet1(z[:,0],z[:,1:].transpose(0,2,3,1)),dw,db
    def cache(model,data,start,end):
        if variant=='literal':return model.forward(data,start,end,True)[1]
        x=data.features;history=[]
        for j in range(len(model.W)-1):
            omega=model.c.siren_first_omega0 if j==0 else model.c.siren_hidden_omega0
            pre=scale1(linear1(x,model.W[j][start:end],model.b[j][start:end]),omega)
            history.append((x,pre,omega));x=old.sine_forward_jet1(pre)
        return x,history
    def reverse(model,data,start,end,history,adj):
        source,layers=history;a=old.envelope_product_backward_jet1(data.envelope,adj)
        a,w,b=linear_reverse(source,a,model.W[-1][start:end]);weights=[w];biases=[b]
        for j in range(len(layers)-1,-1,-1):
            source,pre,omega=layers[j];a=scale1(old.sine_backward_jet1(pre,a),omega)
            a,w,b=linear_reverse(source,a,model.W[j][start:end],variant=='minimal' and j==0)
            weights.append(w);biases.append(b)
        return weights[::-1],biases[::-1]
    def fields(model,data,which='aud',chunk=1024):



        output={key:cp.empty(((data.n if key=='d' else 3*data.n),model.K),data.x.dtype) for key in which}
        for first in range(0,data.n,1024):
            last=min(first+1024,data.n);part=data.part(first,last)
            for start in range(0,model.K,16):
                end=min(start+16,model.K);count=end-start;f,_=model.forward(part,start,end)
                if 'a' in output:output['a'][3*first:3*last,start:end]=f.value.reshape(count,-1).T
                if 'u' in output:output['u'][3*first:3*last,start:end]=curl1(f).reshape(count,-1).T
                if 'd' in output:output['d'][first:last,start:end]=sum(f.grad[...,j,j] for j in range(3)).T
        return output
    def backward(model,data,value=None,curl=None,div=None,grads=None,chunk=1024):
        if grads is None:grads=([cp.zeros(w.shape,data.x.dtype) for w in model.W],[cp.zeros(b.shape,data.x.dtype) for b in model.b])
        gw,gb=grads
        for first in range(0,data.n,1024):
            last=min(first+1024,data.n);part=data.part(first,last);n=last-first
            for start in range(0,model.K,16):
                end=min(start+16,model.K);count=end-start;hist=cache(model,part,start,end)
                av=cp.zeros((count,n,3),data.x.dtype);ag=cp.zeros((count,n,3,3),data.x.dtype)
                if value is not None:av[:]=value[3*first:3*last,start:end].T.reshape(count,n,3)
                if curl is not None:ag+=curl_adjoint(curl[3*first:3*last,start:end].T.reshape(count,n,3).astype(data.x.dtype))
                if div is not None:
                    for j in range(3):ag[...,j,j]+=div[first:last,start:end].T
                dw,db=reverse(model,part,start,end,hist,Jet1(av,ag))
                for j in range(len(gw)):gw[j][start:end]+=dw[j];gb[j][start:end]+=db[j]
        return grads
    grouped.network_fields=fields;grouped.network_backward=backward
    def restore():grouped.network_fields=original_fields;grouped.network_backward=original_backward
    cm._selected_runtime_restore=[restore]
    return dict(name='New velocity-matched nativeCuPy Jet1 '+variant,mode_group=16,point_chunk=1024,
        ANN='FP32',parameter_gradient_accumulation='FP32,1024-point sequential chunks',constraints_Gram_objective='FP64',
        local_parameter_products='Velocity-style packed batched matmul',precision_settings_matched_to_velocity=True,
        original_physical_Potential_objective=True,qualification='Pending this new configuration actual complete preflight',
        qualification_is_not_inherited_from_seed0=True,CUDA_graph=False,optimizer_changed=False)
