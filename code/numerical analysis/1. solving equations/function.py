def judge(x):
    ''''Determine the number of decimal places based on the tolerance.'''
    num = 0
    while x<1:
        x = 10*x
        num += 1
    return num