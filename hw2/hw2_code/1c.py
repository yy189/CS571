from sympy import *

k = Symbol('k')
x = Symbol('x')
m = integrate(sqrt(x)-k*x, (x, 0, 1/k**2)) + integrate(k*x-sqrt(x), (x, 1/k**2, 1)) - 1/2*(1-1/k)*(k-1)
s = solve(diff(m))
print(s[0].evalf())

k=s[0]
m = integrate(sqrt(x)-k*x, (x, 0, 1/k**2)) + integrate(k*x-sqrt(x), (x, 1/k**2, 1))
print(m.evalf())
