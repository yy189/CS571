from sympy import *

k = Symbol('k')
x = Symbol('x')
m = integrate(k*x, (x, 0, 0.5)) + integrate(0.25-k*x, (x, 0.5, 0.25/k)) + integrate(k*x-0.25, (x, 0.25/k, 1))
s = solve(diff(m))
print(s[1].evalf())

k=s[1]
m = integrate(k*x, (x, 0, 0.5)) + integrate(0.25-k*x, (x, 0.5, 0.25/k)) + integrate(k*x-0.25, (x, 0.25/k, 1))
print(m.evalf())
