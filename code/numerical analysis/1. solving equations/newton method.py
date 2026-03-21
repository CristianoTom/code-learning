import sympy as sp

def newton_method(efxpr, x0, tol=1e-10, max_iter=100):
	'''Find a root of the equation f(x) = 0 using the Newton method.\n
    efxpr: the expression for f(x) as a sympy expression.\n
    x0: the initial guess for the root.\n
    tol: the tolerance for convergence (default is 1e-10).\n
    max_iter: the maximum number of iterations (default is 100).\n
    Returns a tuple containing the approximate root, the number of iterations, and the derivative expression.
	'''
	x = sp.symbols("x")
	dexpr = sp.diff(efxpr, x)
	f = sp.lambdify(x, efxpr, "numpy")
	df = sp.lambdify(x, dexpr, "numpy")

	xn = float(x0)
	for i in range(max_iter):
		dfx = float(df(xn))
		if abs(dfx) < 1e-14:
			raise ZeroDivisionError("Derivative is too close to zero; Newton method failed.")
		xn_new = xn - float(f(xn)) / dfx
		if abs(xn_new - xn) < tol:
			return xn_new, i+1, dexpr
		xn = xn_new
	raise RuntimeError("Newton method did not converge within max_iter.")



x = sp.symbols("x")

# Only edit this expression if you want to solve another equation f(x)=0.
expr = x**3 - x - 2
initial_guess = 1.5

root, steps, dexpr = newton_method(expr, initial_guess)

print(f"f(x) = {expr}")
print(f"f'(x) = {dexpr}")
print(f"Approximate root: {root:.12f}")
print(f"f(root) = {float(expr.subs(x, root)):.3e}")
print(f"Iterations: {steps}")
