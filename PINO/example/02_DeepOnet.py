# 调用库
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import torch
import sys
import matplotlib.pyplot as plt
ROOT = Path.cwd()
if not (ROOT / "tutorial_utils.py").exists():
    ROOT = ROOT / "PINO/example/tutorial_utils.py"
sys.path.insert(0, str(ROOT))
from tutorial_utils import *
seed_all()
print("device:", DEVICE, "| smoke:", SMOKE)


### 
ntrain,ntest = counts()
sensors = torch.linspace(0,1,33)
query = torch.linspace(0,1,33)
atrain,aval,atest = torch.randn(ntrain,5),torch.randn(ntest,5),torch.randn(ntest,5)
ftrain,_ = poisson_data(atrain,sensors)
fval,_ = poisson_data(aval,sensors)
ftest,_ = poisson_data(atest,sensors)
_,vtrain = poisson_data(atrain,query)
_,vval = poisson_data(aval,query)
_,vtest = poisson_data(atest,query)
print(ftrain)


### 
class DeepONet(nn.Module):
    def __init__(self, nsensors, latent = 32):
        super().__init__()
        self.branch = nn.Sequential(nn.Linear(nsensors, 64), nn.Tanh(), nn.Linear(64, latent))
        self.trunk = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), 
                                   nn.Linear(64, 64), nn.Tanh(),
                                   nn.Linear(64, latent))

    def forward(self, f, x):
        b = self.branch(f)
        t = self.trunk(x[:, None])
        return (b@t.T)*(4*x*(1-x))[None, :]

model = DeepONet(len(sensors))
predict = lambda m, f: m(f, query.to(f.device))
history = fit(model, ftrain, vtrain, fval, vval, steps = epochs(600), predict = predict)
plot_history(history)


### 
dense = torch.linspace(0, 1, 129)
_, truth = poisson_data(atest, dense)
dense_predict = lambda m, f : m(f, dense.to(f.device))
pred, err = report(model, ftest, truth, predict = dense_predict)
plot_1d(dense, truth / torch.pi**2, pred / torch.pi**2)
assert pred[:, [0, -1]].abs().max() < 1e-7
print("33 sensors - >", pred.shape[-1], "query points")


### 
random_query = torch.rand(75).sort().values
_, random_truth = poisson_data(atest, random_query)
random_pred, _ = report(model, ftest, random_truth,
                        predict = lambda m, f :m(f, random_query.to(f.device)))
plot_1d(random_query, random_truth / torch.pi**2, random_pred / torch.pi**2)

# predict = lambda m, f, r: m(f, r.to(f.device))



