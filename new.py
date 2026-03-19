import math
def rhoo(T):    # density
    return 3.20539 - 0.00962 * T + 9.55357e-6 * T **2 
def lambb(T):   # thermal conductive coefficent
    return 0.00727 + 5.62879e-5 * T + 2.23214e-8 * T**2
def nuu(T):     # dynamic viscosity
    return -1.13502e-6 + 8.823987e-8 * T - 5.60362e-11 * T**2

def Dhh(V, A):  # Hydraulic diameter
    return 4*V/A
def Ree(T, dh, v):     # Reynolds number
    return rhoo(T)*dh*v/nuu(T)
def Nuu(h, dh, T):     #  Nusselt number  
    return h*dh/lambb(T)

def h_10(t_out, t_in, tpms) :
    return (t_out - t_in) * 2 /(2 * tpms - t_in - t_out)
def h_20(t_out, t_in, tpms) :
    delta_T1 = tpms - t_in
    delta_T2 = tpms - t_out    
    return (t_out - t_in) /(tpms - (delta_T1 - delta_T2)/math.log(delta_T1 / delta_T2))
def h_1(rho, a0, v, cp, tpms, h10, a1):   # saunshu  
    return rho * v * a0 *cp * h10/a1
def h_2(rho, a0, v, cp, tpms, h20, a1):   # duishu 
    return rho * v * a0 *cp * h20/a1

def solve(v, tpms, t_out, V = 0, A = 1):
    a0 = 1e-4
    a1 = 4e-4
    t_in = 293.15
    cp = 1010
    T = (t_in + t_out)/2
    rho = rhoo(T)
    Dh = Dhh(V, A)
    h10 = h_10(t_out, t_in, tpms)
    h20 = h_20(t_out, t_in, tpms)
    h1 = h_1(rho, a0, v, cp, tpms, h10, a1)
    h2 = h_2(rho, a0, v, cp, tpms, h20, a1)
    Re = Ree(T, Dh, v)
    Nu1 =  Nuu(h1, Dh, T)
    Nu2 =  Nuu(h2, Dh, T)
    return h1, h2, Nu1, Nu2



outcome1 = solve(3, 361.74, 323.97, 3.1102E-6, 0.0055170)
print(outcome1)
outcome2 = solve(3, 361.81, 323.92, 3.0973E-6, 0.0056909)
print(outcome2)
outcome3 = solve(3, 360.51, 324.35, 3.0720E-6, 0.0059308)
print(outcome3)
outcome4 = solve(3, 360.38, 324.6, 3.0797E-6, 0.0059308)
print(outcome4)
print(outcome3[0]/outcome1[0])


