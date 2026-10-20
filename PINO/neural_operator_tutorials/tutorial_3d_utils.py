"""Grid, synthetic fields and plots for advanced 3D lessons."""
import torch
import matplotlib.pyplot as plt

# Includes negative components and genuine dependence on all three spatial axes.
WAVEVECTORS = ((1,0,0),(0,1,0),(0,0,1),(1,1,0),
               (1,0,1),(0,1,1),(1,-1,1),(2,1,-1))
NCOEFF = 2*len(WAVEVECTORS)

def periodic_grid3d(shape,device=None,dtype=torch.float32):
    shape = (shape,)*3 if isinstance(shape,int) else tuple(shape)
    axes = [torch.arange(n,device=device,dtype=dtype)/n for n in shape]
    return torch.stack(torch.meshgrid(*axes,indexing="ij"))

def field3d(coeff,shape):
    grid = periodic_grid3d(shape,device=coeff.device,dtype=coeff.dtype)
    k = torch.tensor(WAVEVECTORS,device=coeff.device,dtype=coeff.dtype)
    angles = 2*torch.pi*torch.einsum("pc,cxyz->pxyz",k,grid)
    norm2 = k.square().sum(-1)
    basis = torch.stack([angles.sin(),angles.cos()],dim=1).flatten(0,1)
    return torch.einsum("bp,pxyz->bxyz",coeff/norm2.repeat_interleave(2),basis)

def laplace_symbol3d(shape,device=None,dtype=torch.float32):
    nx,ny,nz = shape
    kx = 2*torch.pi*torch.fft.fftfreq(nx,d=1/nx,device=device,dtype=dtype)
    ky = 2*torch.pi*torch.fft.fftfreq(ny,d=1/ny,device=device,dtype=dtype)
    kz = 2*torch.pi*torch.fft.rfftfreq(nz,d=1/nz,device=device,dtype=dtype)
    return kx[:,None,None].square()+ky[None,:,None].square()+kz[None,None,:].square()

def screened_poisson3d(f):
    lam = laplace_symbol3d(f.shape[-3:],f.device,f.dtype)
    return torch.fft.irfftn(torch.fft.rfftn(f,dim=(-3,-2,-1))/(1+lam),
                           s=f.shape[-3:],dim=(-3,-2,-1))

def heat3d(initial,nu,time):
    lam = laplace_symbol3d(initial.shape[-3:],initial.device,initial.dtype)
    nu = torch.as_tensor(nu,device=initial.device,dtype=initial.dtype).reshape(-1,1,1,1)
    time = torch.as_tensor(time,device=initial.device,dtype=initial.dtype).reshape(-1,1,1,1)
    decay = torch.exp(-nu*time*lam)
    return torch.fft.irfftn(torch.fft.rfftn(initial,dim=(-3,-2,-1))*decay,
                           s=initial.shape[-3:],dim=(-3,-2,-1))

def show_volume_slices(truth,pred,index=0):
    truth,pred = truth[index].detach().cpu(),pred[index].detach().cpu()
    error = pred-truth
    vmax = max(truth.abs().max().item(),pred.abs().max().item(),1e-8)
    emax = max(error.abs().max().item(),1e-8)
    fig,axes = plt.subplots(2,3,figsize=(11,6))
    nx,ny,nz = truth.shape
    for row in range(2):
        for col,(volume,title,limit) in enumerate(
            zip([truth,pred,error],["Reference","Prediction","Error"],[vmax,vmax,emax])):
            panel = volume[:,:,nz//2] if row==0 else volume[nx//2,:,:]
            ax = axes[row,col]
            im = ax.imshow(panel.numpy(),origin="lower",extent=[0,1,0,1],
                           cmap="coolwarm",vmin=-limit,vmax=limit)
            ax.set_title(title+(" / z=1/2" if row==0 else " / x=1/2"))
            ax.set_xlabel("y" if row==0 else "z")
            ax.set_ylabel("x" if row==0 else "y")
            ax.grid(False); fig.colorbar(im,ax=ax)
    plt.tight_layout(); plt.show()

def heat_at_queries(coeff,queries,nu=0.04):
    """Exact finite-mode heat solution at arbitrary (x,y,z,t); [B,Q]."""
    k = torch.tensor(WAVEVECTORS,device=queries.device,dtype=queries.dtype)
    norm2 = k.square().sum(-1)
    angles = 2*torch.pi*(queries[:,:3]@k.T)
    spatial = torch.stack([angles.sin(),angles.cos()],dim=-1).flatten(1,2)
    decay = torch.exp(-nu*4*torch.pi**2*queries[:,3,None]*norm2[None,:])
    basis = spatial*decay.repeat_interleave(2,dim=1)/norm2.repeat_interleave(2)
    return coeff@basis.T

