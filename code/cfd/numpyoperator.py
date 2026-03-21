import timeit
import numpy as np
# use record time module
def jishi(f):
    t = timeit.timeit(f, number=10)
    return "{:.3e} s per loop".format(t/10)

# set up initial conditions
nx = 81
ny = 81
nt = 100
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .2
dt = sigma * dx

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

u = np.ones((ny, nx)) 
un = np.ones((ny, nx)) 

# use for loop to update u
def f1():
    u = np.ones((ny, nx))
    u[int(.5 / dy): int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2

    for n in range(nt + 1): ##loop across number of time steps
        un = u.copy()
        row, col = u.shape
        for j in range(1, row):
            for i in range(1, col):
                u[j, i] = (un[j, i] - (c * dt / dx * 
                                    (un[j, i] - un[j, i - 1])) - 
                                    (c * dt / dy * 
                                    (un[j, i] - un[j - 1, i])))
                u[0, :] = 1
                u[-1, :] = 1
                u[:, 0] = 1
                u[:, -1] = 1
    return 0

# use numpy array to update u
def f2():
    u = np.ones((ny, nx))
    u[int(.5 / dy): int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2

    for n in range(nt + 1): ##loop across number of time steps
        un = u.copy()
        u[1:, 1:] = un[1:, 1:] - ((c * dt / dx * (un[1:, 1:] - un[1:, 0:-1])) -
                                  (c * dt / dy * (un[1:, 1:] - un[0:-1, 1:])))
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1
    return 0

# showing the time taken for each function
print("Time taken by for loop:", jishi(f1))
print("Time taken by numpy array:", jishi(f2))