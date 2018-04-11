from sympy import *

x = Symbol('x')
s1 = Symbol('s1')
s2 = Symbol('s2')
s3 = Symbol('s3')
m = integrate(sqrt(x), (x, 0, s2)) + integrate(s1-sqrt(x), (x, s2, s1**2)) + integrate(sqrt(x)-s1, (x, s1**2, s3)) + integrate(1-sqrt(x), (x, s3, 1))
pprint(m)
