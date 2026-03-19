import math
def judge(x):
    num = 0
    while x<1:
        x = 10*x
        num += 1
    return num 
theta = 1e-20


# main function
def FPI(f, x0, tolerance):
    '''
    Find a root of the function f using fixed-point iteration.
    f: the function for which we are trying to find a root
    x0: the initial guess for the root
    tolerance: the desired accuracy of the result
    '''
    def g(x):
        return x + f(x)
    r =  x0
    rn = g(x0)
    while  abs(rn - r)/max(rn,theta) > tolerance:
        r = rn
        rn = g(r)
    if f(rn) < 1e-6:
        return round(rn, judge(tolerance))
    else:
        return 'error'

def f(x):
    return math.cos(x) - x
print(f"Root: {FPI(f, 0, 1e-6)}")
