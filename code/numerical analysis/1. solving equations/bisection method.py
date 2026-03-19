import math
def judge(x):
    num = 0
    while x<1:
        x = 10*x
        num += 1
    return num - 1

def bisection_method(f, a, b, tolerance):
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
    return round((a + b) / 2, judge((b - a) / 2)), num


# main function
def f(x):
    return math.cos(x) - x #x**3 + x - 1
root, number = bisection_method(f, 0, 1, 1e-6)
print(f"Root: {root}, Number of iterations: {number}")
