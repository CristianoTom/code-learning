import math
from function import judge

def bisection_method(f, a, b, tolerance):
    '''
    Find a root of the function f in the interval [a, b] using the bisection method.\n
    f: the function for which we are trying to find a root.\n
    a: the start of the interval.\n
    b: the end of the interval.\n
    tolerance: the desired accuracy of the result.
    '''
    if f(a) * f(b) >= 0:
        print("The function must have opposite signs at the endpoints.")
        return None
    num = 0
    while (b - a) / 2 > tolerance:
        c = (a + b) / 2
        if f(c) == 0:
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        num += 1
    return round((a + b) / 2, judge(tolerance)), num


# main function
def f(x):
    return math.cos(x) - x #x**3 + x - 1
root, number = bisection_method(f, 0, 1, 1e-6)
print(f"Root: {root}, Number of iterations: {number}")
