from pure_newton_robust import pure_newton
import numpy as np
from gradient_method_backtracking import gradient_method_backtracking
f = lambda x: 100*x[0]**4 + 0.01*x[1]**4
g = lambda x: np.array([400*x[0]**3, 0.04*x[1]**3])

#h = lambda x: np.array([[1200*x[0]**2, 0],[0, 0.12*x[1]**2]])
h = lambda x: np.array([[1200*x[0]**2, 0.0],[0.0, 0.12*x[1]**2]])


x0 = np.array([1,1])
epsilon = 10**-6

s = 1
alpha = 0.5
beta = 0.5


#Converge luego de 1612 iteraciones,
#Esta mal escalada
gradient_method_backtracking(f,g,x0,s,alpha,beta,epsilon)


#Converge luego de 17 iteraciones
pure_newton(f,g,h,x0,epsilon)


#Calculo numero de condicionamiento
#print(np.linalg.eig(h(x0)))

eignvalues = np.linalg.eig(h(x0))[0]

#Numero de condicionamiento muy grande!!!

print(f"El numero de condicionamiento es: {eignvalues[0]/eignvalues[1]}")