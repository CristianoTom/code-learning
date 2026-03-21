import math
from function import judge

theta = 1e-10
# main function
def FPI(f, x0, tolerance):
    '''
    Find a root of the function f using fixed-point iteration.\n
    f: the function for which we are trying to find a root.\n
    x0: the initial guess for the root.\n
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
