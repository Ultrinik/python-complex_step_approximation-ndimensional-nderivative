import numpy as np
from derivative_approximations import get_csa_function, get_fdf_function
import timeit

def main():
    cpoints = [1]
    cpoints2 = [1,2,3]
    fpoints = [0,1]
    fpoints2 = [-2,-1,0,1,2]

    csa_func, _, _ = get_csa_function(cpoints, 1, False)
    csa2_func, _, _ = get_csa_function(cpoints2, 1, False)
    fdf_func, _, _ = get_fdf_function(fpoints, 1, False)
    fdf2_func, _, _ = get_fdf_function(fpoints2, 1, False)

    x = np.array([1., 2.])
    v = np.array([1., 1.])
    h = 1e-8

    def func(x):
        return np.sin(x)+np.cos(x)
    def derivate_func(x):
        return np.cos(x)-np.sin(x)

    times = 1000

    rr = derivate_func(x)
    print("Real result:              ", rr)
    print("")

    r = csa_func(func, x, v, h)
    print("Complex Step error:       ", r-rr, np.linalg.norm(r-rr))
    t = timeit.Timer(lambda: csa_func(func, x, v, h))
    print("Execution Time:           ", t.timeit(times)/times)

    r = csa2_func(func, x, v, h)
    print("Complex Step 2 error:     ", r-rr, np.linalg.norm(r-rr))
    t = timeit.Timer(lambda: csa2_func(func, x, v, h))
    print("Execution Time:           ", t.timeit(times)/times)
    
    print("")

    r = fdf_func(func, x, v, h)
    print("Finite Difference error:  ", r-rr, np.linalg.norm(r-rr))
    t = timeit.Timer(lambda: fdf_func(func, x, v, h))
    print("Execution Time:           ", t.timeit(times)/times)

    r = fdf2_func(func, x, v, h)
    print("Finite Difference 2 error:", r-rr, np.linalg.norm(r-rr))
    t = timeit.Timer(lambda: fdf2_func(func, x, v, h))
    print("Execution Time:           ", t.timeit(times)/times)

if __name__ == "__main__":
    main()