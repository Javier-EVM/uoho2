from newton_backtracking import newton_backtracking
from pure_newton_robust import pure_newton
import numpy as np
from gradient_method_backtracking import gradient_method_backtracking

f = lambda x: np.sqrt(x[0]**2 + 1) + np.sqrt(x[1]**2 + 1)
g = lambda x: np.array([ x[0]/(np.sqrt(x[0]**2 + 1)), x[1]/(np.sqrt(x[1]**2 + 1))])

#h = lambda x: np.array([[1200*x[0]**2, 0],[0, 0.12*x[1]**2]])
h = lambda x: np.array([ [1 / ((x[0]**2 + 1)**(3/2)), 0.0],[0.0, 1 / ( (x[1]**2 + 1)**(3/2) )] ])


#Newton converge 0.5 0.5
#x0 = np.array([0.5,0.5])


#Newton no converge 1, 1
x0 = np.array([1,1])
s = 1
alpha = 0.5
beta = 0.5
epsilon = 10**-8



#pure_newton(f,g,h,x0,epsilon)


gradient_method_backtracking(f,g,x0,s,alpha,beta,epsilon)


newton_backtracking(f,g,h,x0,alpha,beta,epsilon)