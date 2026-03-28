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



outcome1 = solve(0.5, 371.07, 365.45, 7.1038E-6, 0.011889)
print(outcome1)
outcome2 = solve(1, 369.43, 359.36, 7.1038E-6, 0.011889)
print(outcome2)
outcome3 = solve(1.5, 367.96, 355.4, 7.1038E-6, 0.011889)
print(outcome3)
outcome4 = solve(2, 366.59, 352.58, 7.1038E-6, 0.011889)
print(outcome4)
outcome4 = solve(2.5, 365.28, 350.46, 7.1038E-6, 0.011889)
print(outcome4)
outcome1 = solve(3, 364.02, 348.81, 7.1038E-6, 0.011889)
print(outcome1)
outcome2 = solve(3.5, 362.8, 347.46, 7.1038E-6, 0.011889)
print(outcome2)
outcome3 = solve(4, 361.62, 346.31, 7.1038E-6, 0.011889)
print(outcome3)
outcome4 = solve(4.5, 360.48, 345.29, 7.1038E-6, 0.011889)
print(outcome4)
outcome4 = solve(5, 359.38, 344.37, 7.1038E-6, 0.011889)
print(outcome4)
print()



outcome2 = solve(1, 369.41, 358.31, 7.1412E-6, 0.01175)
print(outcome2)

outcome4 = solve(2, 366.6, 351.27, 7.1412E-6, 0.01175)
print(outcome4)

outcome1 = solve(3, 364.8, 347.38, 7.1412E-6, 0.01175)
print(outcome1)

outcome3 = solve(4, 361.76, 344.69, 7.1412E-6, 0.01175)
print(outcome3)

outcome4 = solve(5, 359.6, 342.64, 7.1412E-6, 0.01175)
print(outcome4)
print()
