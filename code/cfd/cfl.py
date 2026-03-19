import numpy                 
import matplotlib.pyplot as plt

def linearconv(nx):
    dx = 2 / (nx - 1)
    nt = 20    
    dt = 0.25
    c = 1

    u = numpy.ones(nx)
    u[int(.5/dx):int(1 / dx + 1)] = 2
    un = numpy.ones(nx) 

    for n in range(nt): 
        un = u.copy() 
        for i in range(1, nx):
            u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])
        
    plt.plot(numpy.linspace(0, 2, nx), u)


def linearconv_right(nx):
    dx = 2 / (nx - 1)
    nt = 20    
    c = 1
    sigma = .5    # Courant number
    
    dt = sigma * dx

    u = numpy.ones(nx) 
    u[int(.5/dx):int(1 / dx + 1)] = 2

    un = numpy.ones(nx)

    for n in range(nt):  
        un = u.copy()
        for i in range(1, nx):
            u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])
        
    plt.plot(numpy.linspace(0, 2, nx), u)