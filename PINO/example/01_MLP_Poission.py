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
ntrain, ntest = counts()
x = torch.linspace(0, 1, 33)
train_a = torch.randn(ntrain, 5)
val_a = torch.randn(ntest, 5)
test_a = torch.randn(ntest, 5)
ftrain, vtrain = poisson_data(train_a, x)
fval, vval = poisson_data(val_a, x)
ftest, vtest = poisson_data(test_a, x)
print(ftrain.shape, vtrain.shape)

###
class GridMLP(nn.Module):
    def __init__(self, grid):
        super().__init__()
        n = len(grid)
        self.net = nn.Sequential(
            nn.Linear(n, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, n)
        )
        self.register_buffer("boundary", 4*grid*(1- grid))
    def forward(self, f):
        return self.net(f)*self.boundary

model = GridMLP(x)
print("parameters:", sum(p.numel() for p in model.parameters()))
history = fit(model, ftrain, vtrain, fval, vval, steps = epochs(1800))
plot_history(history)


pred, error = report(model, ftest, vtest)
plot_1d(x, vtest / torch.pi**2, pred / torch.pi**2)
plt.hist(error.numpy(), bins = 10)
plt.title("Test relative L2 distribution")
plt.show()

assert pred[:, [0,-1]].abs().max() < 1e-7
print("boundary max error:", pred[:, [0,-1]].abs().max().item())



