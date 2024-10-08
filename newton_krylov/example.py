import numpy as np
from scipy.optimize import newton_krylov
from newton_krylov_csa import newton_krylov_csa
import timeit

def main():

    def _func(x):
        return x**2 - np.sin(x) * np.cos(x)
    
    xr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    n = len(xr)

    def func(x):
        return _func(x) - _func(xr)

    x0 = np.zeros(n)
    newton_iters = 100
    inner_maxiter = 20
    tol = 0.0
    h = 1e-12

    times = 100

    print("Original Solution:          ", xr)
    
    print("")

    solution = newton_krylov(func, x0, method="gmres", inner_maxiter=inner_maxiter, iter=newton_iters, f_tol=tol, rdiff=h)
    print("Default Scipy Solution:     ", solution, "Error:", np.linalg.norm(func(solution)))
    t = timeit.Timer(lambda: newton_krylov(func, x0, method="gmres", inner_maxiter=inner_maxiter, iter=newton_iters, f_tol=tol, rdiff=h))
    print("Execution Time:             ", t.timeit(times)/times)

    print("")

    solution = newton_krylov_csa(func, x0, method="gmres", inner_maxiter=inner_maxiter, iter=newton_iters, f_tol=tol, rdiff=h)
    print("New Implementation Solution:", solution, "Error:", np.linalg.norm(func(solution)))
    t = timeit.Timer(lambda: newton_krylov_csa(func, x0, method="gmres", inner_maxiter=inner_maxiter, iter=newton_iters, f_tol=tol, rdiff=h))
    print("Execution Time:             ", t.timeit(times)/times)

if __name__ == "__main__":
    main()