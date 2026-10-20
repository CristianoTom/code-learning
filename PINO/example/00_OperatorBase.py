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


# 正弦基傅里叶算子
x = torch.linspace(0, 1, 65)
a = torch.randn(4, 5)
f, v = poisson_data(a, x)
u = v / torch.pi ** 2
fig, ax = plt.subplots(1, 2, figsize = (10, 3))
ax[0].plot(x, f.T)
ax[0].set_title("Input Forcing f")
ax[1].plot(x, u.T)
ax[1].set_title("Output Solution u")
plt.tight_layout()
plt.show()


# 中心差分
def finite_difference(f):
    n = f.shape[-1]
    h = 1 / (n - 1)
    m = n - 2
    A = (2* torch.eye(m, dtype = f.dtype) 
         - torch.diag(torch.ones(m-1, dtype = f.dtype), 1)
         - torch.diag(torch.ones(m-1, dtype = f.dtype), -1)) / h**2
    interior = torch.linalg.solve(A, f[:, 1: -1].T).T
    return torch.nn.functional.pad(interior, (1, 1))

errors = []
for n in [17, 33, 65, 129]:
    grid = torch.linspace(0, 1, n)
    forcing, scaled = poisson_data(a.double(), grid.double())
    exact = scaled / torch.pi ** 2
    numerical = finite_difference(forcing.double())
    err = relative_l2(numerical, exact).mean().item()
    errors.append(err)
    print("N = {}, relative L2 = {:.6f}".format(n, err))
assert errors[-1] < errors[0]
fd = finite_difference(f)
plot_1d(x, u, fd)
plt.show()





