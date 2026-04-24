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
def h_20(t_out, t_in, tpms) :
    delta_T1 = tpms - t_in
    delta_T2 = tpms - t_out    
    return (t_out - t_in) /(tpms - (delta_T1 - delta_T2)/math.log(delta_T1 / delta_T2))
def h_1(rho, a0, v, cp, tpms, h10, a1):   # saunshu  
    return rho * v * a0 *cp * h10/a1
def h_2(rho, a0, v, cp, tpms, h20, a1):   # duishu 
    return rho * v * a0 *cp * h20/a1

def solve(v, tpms, t_out, V = 0, A = 1):
    a0 = 4e-4
    a1 = 16e-4
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
    return h1, h2, Nu1, Nu2, Re



outcome1 = solve(0.5, 370.05 , 361.22, 4.5328E-6, 0.011013)
print(outcome1)
outcome2 = solve(1, 367.77 , 353.48, 4.5328E-6, 0.011013)
print(outcome2)
outcome3 = solve(1.5,365.86 , 348.45,4.5328E-6, 0.011013)
print(outcome3)
outcome4 = solve(2, 364.16 , 345.09, 4.5328E-6, 0.011013)
print(outcome4)
outcome4 = solve(2.5, 362.61 , 342.39, 4.5328E-6, 0.011013)
print(outcome4)
outcome1 = solve(3, 361.2 , 340.37, 4.5328E-6, 0.011013)
print(outcome1)
outcome2 = solve(3.5, 361.41, 348, 4.5328E-6, 0.011013)
print(outcome2)
outcome3 = solve(4, 358.6, 336.95, 4.5328E-6, 0.011013)
print(outcome3)
outcome4 = solve(4.5, 357.35,335.78, 4.5328E-6, 0.011013)
print(outcome4)
outcome4 = solve(5,356.15, 334.82, 4.5328E-6, 0.011013)
print(outcome4)
print()



outcome2 = solve(3, 362.5, 341.35, 6.367E-6, 0.011137)
print(outcome2)

outcome4 = solve(3, 362.3, 342.42, 6.5773E-6, 0.011754)
print(outcome4)

outcome1 = solve(3.5,	362.52	,339.85	,6.4979e-6	,0.011025)
print(outcome1)

outcome3 = solve(4.5,	359.16	,337.33	,6.4979e-6	,0.011025)
print(outcome3)

outcome4 = solve(3,	362.36	,342.42	,6.5773e-6	,0.011754)

print(outcome4)
print()
