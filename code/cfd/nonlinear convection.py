import numpy as np
import matplotlib.pyplot as plt

# set grid conditions
nx = 41
dx = 2 / (nx - 1)
nt = 25
dt = 0.025
c = 1

# set initial conditions
u = np.ones(nx)
u[int(0.5 / dx):int(1 / dx + 1)] = 2

# begin calculate
un = np.ones(nx)
for i in range(nt):
    un = u.copy()
    for j in range(1, nx):
        u[j] = un[j] - un[j] * dt / dx * (un[j] - un[j - 1])

plt.plot(np.linspace(0, 2, nx), u)
plt.show()