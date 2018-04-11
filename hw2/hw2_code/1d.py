from sympy import *

x = Symbol('x')
s1 = Symbol('s1')
s2 = Symbol('s2')
s3 = Symbol('s3')
m = integrate(s2-sqrt(x), (x, 0, s2**2)) + integrate(sqrt(x)-s2, (x, s2**2, s1)) + integrate(s3-sqrt(x), (x, s1, s3**2)) + integrate(sqrt(x)-s3, (x, s3**2, 1))
pprint(m)



