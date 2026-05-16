import math
def rhoo(T):    # density
    return 3.20539 - 0.00962 * T + 9.55357e-6 * T **2 
def lambb(T):   # thermal conductive coefficent
    return 0.00727 + 5.62879e-5 * T + 2.23214e-8 * T**2
def nuu(T):     # dynamic viscosity
    return -1.13502e-6 + 8.823987e-8 * T - 5.60362e-11 * T**2

def Dhh(V, A):  # Hydraulic diameter
    return (32*1e-6-V)*4/A
def Ree(T, dh, v):     # Reynolds number
    return rhoo(T)*dh*v/nuu(T)
def Nuu(h, dh, T):     #  Nusselt number  
    return h*dh/lambb(T)

def h_10(t_out, t_in, tpms) :
    return (t_out - t_in) * 2 /(2 * tpms - t_in - t_out)
def h_1(rho, a0, v, cp, h10, a1):   # saunshu  
    return rho * v * a0 *cp * h10/a1

def ff(v, Dh, p, rho, L=0.08):
    return p*Dh/2/rho/v**2/L


def solve(v, tpms, t_out, V , A , p):
    V = V*1e-6
    a0 = 4e-4
    a1 = 16e-4
    t_in = 293.15
    cp = 1010
    T = (t_in + t_out)/2
    rho = rhoo(T)
    Dh = Dhh(V, A)
    h10 = h_10(t_out, t_in, tpms)
    h1 = h_1(rho, a0, v, cp, h10, a1)
    Re = Ree(T, Dh, v)
    Nu1 =  Nuu(h1, Dh, T)
    f = ff(v, Dh, p, rho)
    return h1, Nu1, Re, f

def pec(l1, l2):
    return l2[1]/l1[1]/(l2[3]/l1[3])**(1/3)

def pec1(l1, l2,l3, l4):
    return l1/l2/(l3/l4)**(1/3)


outcome1 = solve(3, 360.37, 344.57, 5.7848e-6,0.012887,214)
print(outcome1)
outcome2 = solve(3, 363.72, 342.92, 6.00051e-6,0.010886,310)
print(outcome2)
outcome3 = solve(3,	360.83	,344.06	,5.8239e-6	,0.012648,205)
print(outcome3)
outcome3 = solve(3,	362.33	,347.53	,5.8289e-6	,0.012657,392)
print(outcome3)



outcome1 = solve(0.5, 370.4, 363.62, 6.3983e-6, 0.012149,9)
outcome2 = solve(1.0, 368.37, 356.17, 6.3983e-6, 0.012149,28)
outcome3 = solve(1.5, 366.59, 351.51, 6.3983e-6, 0.012149,57)
outcome4 = solve(2.0, 365.06, 347.88, 6.3983e-6, 0.012149,95)
outcome5 = solve(2.5, 363.66, 344.76, 6.3983e-6, 0.012149,140)
outcome6 = solve(3.0, 362.36, 342.42, 6.3983e-6, 0.012149,196)
print(outcome1, outcome2, outcome3, outcome4, outcome5, outcome6, sep='\n')

print()

outcome1 = solve(0.5, 370.23, 364.22, 6.3983, 0.012149, 9.26)
outcome2 = solve(1.0, 368.04, 357.06, 6.3983, 0.012149, 29.37)
outcome3 = solve(1.5, 366.14, 352.51, 6.3983, 0.012149, 59.00)
outcome4 = solve(2.0, 364.49, 348.65, 6.3983, 0.012149, 98.00)
outcome5 = solve(2.5, 362.99, 345.74, 6.3983, 0.012149, 146.00)
outcome6 = solve(3, 362.42, 340.94, 6.2184, 0.010907, 183.61)

print(outcome1, outcome2, outcome3, outcome4, outcome5,
      outcome6, sep='\n')
