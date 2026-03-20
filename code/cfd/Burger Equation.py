import numpy as np       
import matplotlib.pyplot as plt
import sympy as sp  
from sympy import init_printing 

x, nu, t = sp.symbols('x nu t')
phi = (sp.exp(-(x - 4 * t)**2 / (4 * nu * (t + 1))) +
       sp.exp(-(x - 4 * t - 2 * sp.pi)**2 / (4 * nu * (t + 1))))
phiprime = phi.diff(x)  

from sympy.utilities.lambdify import lambdify
u = -2 * nu * (phiprime / phi) + 4

###variable declarations
nx = 101
nt = 100
dx = 2 * numpy.pi / (nx - 1)
nu = .07
dt = dx * nu

x = np.linspace(0, 2 * np.pi, nx)
un = np.empty(nx)
t = 0

u = np.asarray([ufunc(t, x0, nu) for x0 in x])
u


