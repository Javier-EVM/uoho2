import numpy as np

def gradient_scaled_quadratic(A,b,D,x0,epsilon):
    """
    % INPUT
    % ======================
    % A ....... the positive definite matrix associated
    % with the objective function
    % b ....... a column vector associated with the linear part
    % of the objective function
    % D ....... scaling matrix
    % x0 ...... starting point of the method
    % epsilon . tolerance parameter
    % OUTPUT
    % =======================
    % x ....... an optimal solution (up to a tolerance)...
    of min(x^T A x+2 b^T x)
    % fun_val . the optimal function value up to a tolerance
    """
    x=x0
    iter=0
    grad=2*(A @ x + b);
    while (np.linalg.norm(grad)>epsilon):
        iter += 1
        t = grad @ D @ grad/( 2*(grad @ D ) @ A @ (D @ grad) )
        x = x - t * D @ grad
        grad = 2*(A @ x+b)
        fun_val = x @ A @ x + 2*b @ x
        print(f"iter_number = {iter} norm_grad = {np.linalg.norm(grad)} fun_val = {fun_val}")
    
    return x, fun_val