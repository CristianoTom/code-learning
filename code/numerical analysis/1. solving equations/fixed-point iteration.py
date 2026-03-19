import math
def judge(x):
    num = 0
    while x<1:
        x = 10*x
        num += 1
    return num 
theta = 1e-20



def f(x):
    return math.cos(x) - math.sin(x)
def FPI(f, x0, tolerance):
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
print(FPI(f, 0, 1e-6))
