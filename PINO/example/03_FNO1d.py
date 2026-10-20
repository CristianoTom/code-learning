# 调用库
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

ntrain, ntest = counts()
x = torch.arange(64) / 64
atrain , aval , atest = torch.randn(ntrain, 10), torch.randn(ntest, 10), torch.randn(ntest, 10)
u0, ut = heat_data(atrain, x)
v0, vt = heat_data(aval, x)
t0, tt = heat_data(atest, x)
plot_1d(x, t0, tt)
print("initial -> final:", u0.shape, ut.shape)





