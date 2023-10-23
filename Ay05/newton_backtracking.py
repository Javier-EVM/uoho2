import numpy as np
def newton_backtracking(f,g,h,x0,alpha,beta,epsilon):
    """
    % Newton’s method with backtracking
    %
    % INPUT
    %=======================================
    % f ......... objective function
    % g ......... gradient of the objective function
    % h ......... hessian of the objective function
    % x0......... initial point
    % alpha ..... tolerance parameter for the stepsize selection strategy
    % beta ...... the proportion in which the stepsize is multiplied
    % at each backtracking step (0<beta<1)
    % epsilon ... tolerance parameter for stopping rule
    % OUTPUT
    %=======================================
    % x ......... optimal solution (up to a tolerance)
    % of min f(x)
    % fun_val ... optimal function value
    """
    x = x0
    gval = g(x)
    hval = h(x)
    d = np.linalg.solve(hval, gval)
    iter = 0
    
    while ((np.linalg.norm(gval)>epsilon) and (iter<10000)):
        iter=iter+1
        t=1
        while(f(x-t*d) > f(x)-alpha*t*gval @ d):
            t=beta*t

        x=x-t*d
        print(f"iter_number = {iter} x = {x}  norm_grad = {np.linalg.norm(g(x))} fun_val = {f(x)}")
        gval = g(x)
        hval = h(x)
        d = np.linalg.solve(hval, gval)

    if (iter==10000):
        print("did not converge\n")
     