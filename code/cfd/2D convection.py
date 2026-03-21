from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# set up initial conditions
nx = 81
ny = 81
nt = 100
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = 0.2
dt = sigma * dx

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

u = np.ones((ny, nx))
# un = np.ones((ny, nx))

## Assign initial conditions
u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2 

###Plot Initial Condition
fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(111, projection='3d')  # create a 3D plot
X, Y = np.meshgrid(x, y)            
surf = ax.plot_surface(X, Y, u[:], cmap=plt.cm.viridis)
plt.show()


# calculate 
for n in range(nt + 1): 
    un = u.copy()
    u[1:, 1:] = (un[1:, 1:] - (c*dt/dx*(un[1:, 1:] - un[1:, :-1])) -
                              (c*dt/dy*(un[1:, 1:] - un[:-1, 1:])))
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

fig = plt.figure(figsize=(11, 7), dpi=100)
ax = plt.subplot(111, projection='3d')  # create a 3D plot
surf1 = ax.plot_surface(X, Y, u[:], cmap=plt.cm.viridis)

plt.show()