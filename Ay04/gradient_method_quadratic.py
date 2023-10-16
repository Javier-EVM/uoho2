import numpy as np

def gradient_method_quadratic(A,b,x0,epsilon):
    """
    % INPUT
    % ======================
    % A ....... the positive definite matrix associated with the
    % objective function
    % b ....... a column vector associated with the linear part of the
    % objective function
    % x0 ...... starting point of the method
    % epsilon . tolerance parameter
    % OUTPUT
    % =======================
    % x ....... an optimal solution (up to a tolerance) of
    % min(x^T A x+2 b^T x)
    % fun_val . the optimal function value up to a tolerance
    """
    x=x0
    x = np.transpose(x)
    iter=0
    grad=2*(A @ x + b)
   
    while (np.linalg.norm(grad)>epsilon):
        iter += 1
        t = (np.linalg.norm(grad)**2) / (2* grad @ A @ grad) #Se calcula el paso de manera exacta
        x = x - t*grad #x_k+1 = x_k -t*grad
        grad = 2 * (A @ x + b) #Re calcula el gradiente para la proxima iteracion
        fun_val = np.transpose(x) @ A @ x + 2*np.transpose(b) @ x
        print(f"iter_number = {iter} norm_grad = {np.linalg.norm(grad)} fun_val = {fun_val}")
    
    return x, fun_val



"""
# Ejemplo de uso
# x^2 + 2y^2
A = np.array([[1, 0], [0, 2]])
b = np.array([0, 0])
x0 = np.array([2, 1])
epsilon = 1e-6

x, fun_val = gradient_method_quadratic(A, b, x0, epsilon)
print(f'Solución óptima: x = {x}, Valor óptimo: {fun_val:.6f}')
"""

# Ejemplo de uso
# 6x^2 + 5y^2 -10x +8y
"""
A = np.array([[6, 0], [0, 5]])
b = np.array([-5, 4])
x0 = np.array([2, -1])
epsilon = 1e-6

x, fun_val = gradient_method_quadratic(A, b, x0, epsilon)
print(f'Solución óptima: x = {x}, Valor óptimo: {fun_val:.6f}')
"""