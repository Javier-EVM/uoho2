import numpy as np
def gradient_method_backtracking(f,g,x0,s,alpha,beta,epsilon):
    """"
    % Gradient method with backtracking stepsize rule
    %
    % INPUT
    %=======================================
    % f ......... objective function
    % g ......... gradient of the objective function
    % x0......... initial point
    % s ......... initial choice of stepsize
    % alpha ..... tolerance parameter for the stepsize selection
    % beta ...... the constant in which the stepsize is multiplied
    % at each backtracking step (0<beta<1)
    % epsilon ... tolerance parameter for stopping rule
    % OUTPUT
    %=======================================
    % x ......... optimal solution (up to a tolerance)
    % of min f(x)
    % fun_val ... optimal function value
    """
    x=x0
    grad=g(x)
    fun_val=f(x)
    iter=0
    while (np.linalg.norm(grad)>epsilon):
        iter=iter+1
        t=s
        while (fun_val-f(x-t*grad) < alpha*t*(np.linalg.norm(grad))**2):
            t=beta*t

        x=x-t*grad
        fun_val=f(x)
        grad=g(x)
        print(f"iter_number = {iter} x = {x} norm_grad = {np.linalg.norm(grad)} fun_val = {fun_val}")
        
    return x,fun_val