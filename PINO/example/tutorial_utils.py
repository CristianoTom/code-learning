"""Small shared utilities. Model and PDE code lives in each notebook."""
### 
import os, random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


### 
SMOKE = os.getenv("NO_TUTORIAL_SMOKE", "0") == "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not SMOKE else "cpu")


### 
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    plt.rcParams.update({
        "figure.figsize": (8, 3), 
        "axes.grid": True
        })

def epochs(default):
    return int(os.getenv(
        "NO_TUTORIAL_EPOCHS", 
        "2" if SMOKE else str(default)
        ))

def counts(train=256, test=64):
    return (32, 8) if SMOKE else (train, test)

def relative_l2(pred, target):
    a = (pred-target).reshape(len(target), -1).norm(dim=1)
    b = target.reshape(len(target), -1).norm(dim=1).clamp_min(1e-8)
    return a/b

def fit(model, x, y, xv, yv, 
        steps=100, lr=0.001, batch=128, predict=None):
    model.to(DEVICE)
    predict = predict or (lambda m,z: m(z))
    loader = DataLoader(TensorDataset(x,y), batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(steps):
        model.train(); total = 0.
        for xb,yb in loader:
            xb,yb = xb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad()
            loss = (predict(model,xb)-yb).square().mean()
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite loss")
            loss.backward()
            opt.step()
            total += loss.item()*len(xb)
        model.eval()
        with torch.no_grad():
            val = relative_l2(predict(model,xv.to(DEVICE)),yv.to(DEVICE)).mean().item()
        history.append((total/len(x),val))
        if epoch == 0 or (epoch+1)%max(1,steps//5)==0:
            print(f"epoch {epoch+1:4d} | train MSE {history[-1][0]:.3e} | val relative L2 {val:.3f}")
    return np.asarray(history)

def report(model,x,y,predict=None):
    predict = predict or (lambda m,z: m(z))
    model.eval()
    with torch.no_grad(): 
        pred = predict(model,x.to(DEVICE)).cpu()
    assert pred.shape==y.shape and torch.isfinite(pred).all()
    errors = relative_l2(pred,y)
    print(f"Test relative L2: mean={errors.mean():.4f}, median={errors.median():.4f}, max={errors.max():.4f}")
    return pred,errors

def plot_history(history):
    fig,ax = plt.subplots(1,2,figsize=(9,3))
    ax[0].semilogy(history[:,0].clip(1e-12)); ax[0].set_title("Train MSE")
    ax[1].plot(history[:,1]); ax[1].set_title("Validation relative L2")
    for a in ax: 
        a.set_xlabel("epoch")
    plt.tight_layout()
    plt.show()
    
def plot_1d(x,target,pred,index=0):
    fig,ax = plt.subplots(1,2,figsize=(10,3))
    ax[0].plot(x,target[index],label="reference")
    ax[0].plot(x,pred[index],"--",label="prediction")
    ax[0].legend()
    ax[0].set_title("Unseen input function")
    ax[1].plot(x,(pred-target)[index])
    ax[1].set_title("Signed error")
    plt.tight_layout(); plt.show()

def poisson_data(coeff,grid):
    """-u_xx=f; returns f and pi^2*u, with zero Dirichlet boundary."""
    k = torch.arange(1,coeff.shape[1]+1,dtype=grid.dtype)
    basis = torch.sin(torch.pi*k[:,None]*grid[None,:])
    return coeff@basis,(coeff/k.square())@basis

def periodic_data(coeff,grid):
    k = torch.arange(1,coeff.shape[1]//2+1,dtype=grid.dtype)
    angle = 2*torch.pi*k[:,None]*grid[None,:]
    basis = torch.cat((angle.sin(),angle.cos()),dim=0)
    return (coeff/torch.cat((k.square(),k.square())))@basis

def heat_data(coeff,grid,nu=0.01,time=0.2):
    k = torch.arange(1,coeff.shape[1]//2+1,dtype=grid.dtype)
    decay = torch.exp(-nu*(2*torch.pi*k).square()*time).repeat(2)
    return periodic_data(coeff,grid),periodic_data(coeff*decay,grid)
