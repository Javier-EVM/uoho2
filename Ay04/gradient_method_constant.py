import numpy as np
def gradient_method_constant(f,g,x0,t,epsilon):
    """   
    % Gradient method with constant stepsize
    %
    % INPUT
    %=======================================
    % f ......... objective function
    % g ......... gradient of the objective function
    % x0......... initial point
    % t ......... constant stepsize
    % epsilon ... tolerance parameter
    % OUTPUT
    %=======================================
    % x ......... optimal solution (up to a tolerance)
    % of min f(x)
    % fun_val ... optimal function value
    """
    x = x0
    grad = g(x)
    iter = 0
    while (np.linalg.norm(grad)>epsilon):
        iter = iter+1
        x = x-t*grad
        fun_val = f(x)
        grad = g(x)
        print(f"iter_number = {iter} norm_grad = {np.linalg.norm(grad)} fun_val = {fun_val}")

    return x,fun_val

"""
#Ejemplo
# 6x^2 + 5y^2 -10x +8y
A = np.array([[6, 0], [0, 5]])
b = np.array([-5, 4])
x0 = np.array([2, -1])
epsilon = 1e-6

#Se definen funciones lambda para f(x) y grad(f(x))
f = lambda x: np.transpose(x) @ A @ x + 2*np.transpose(b) @ x
g = lambda x: 2*(A @ x + b)

#Se define el tamaño del paso
t = 0.1
gradient_method_constant(f,g,x0,t,epsilon)
"""