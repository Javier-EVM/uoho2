import numpy as np
def pure_newton(f,g,h,x0,epsilon):
    """
    % Pure Newton’s method
    %
    % INPUT
    % ==============
    % f .......... objective function
    % g .......... gradient of the objective function
    % h .......... Hessian of the objective function
    86 Chapter 5. Newton’s Method
    % x0........... initial point
    % epsilon ..... tolerance parameter
    % OUTPUT
    % ==============
    % x - solution obtained by Newton’s method (up to some tolerance)
    """

    x = x0
    gval = g(x)
    hval = h(x)
    iter = 0
    while ((np.linalg.norm(gval)>epsilon) and (iter<10000)):
        iter = iter + 1
        d = np.linalg.solve(hval, gval)
        x = x - d
        print(f"iter= {iter} f(x)={f(x)}\n")
        gval=g(x)
        hval=h(x)

    if (iter==10000):
        print("did not converge")
    