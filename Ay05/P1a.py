from pure_newton_robust import pure_newton
import numpy as np

f = lambda x: np.sqrt(1 + x**2)
g = lambda x: x/np.sqrt(1 + x**2)
h = lambda x: 1/np.sqrt(1 + x**2)**(3/2)

#hasta 3 funciona Converge

#4 diverge
x0 = 3
epsilon = 10**-5
pure_newton(f,g,h,x0,epsilon)