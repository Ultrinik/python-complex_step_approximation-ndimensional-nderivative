import numpy as np
from multicomplex import Multicomplex
from derivative_approximations import get_csa_function, get_fdf_function

def main():

    x = np.array([1., 2., 3.])
    v = np.array([1., 1., 1.])
    h = 1e-8

    def func_0(x):
        return np.sin(x)
    def func_1(x):
        return np.cos(x)
    def func_2(x):
        return -np.sin(x)
    def func_3(x):
        return -np.cos(x)
   
    funcs = [func_0, func_1, func_2, func_3]
    for i in range(1,10):

        print("Order:", i)

        equivalent_order = i%4
        rr = funcs[equivalent_order](x)
        print("Real result:              ", rr)

        r = Multicomplex.complex_step(func_0, x, v, h, i)
        print("Multicomplex Step error:  ", r-rr, np.linalg.norm(r-rr))


        points = list(range(i+1))
        csa_func, _, _ = get_csa_function(points, i, False)
        r = csa_func(func_0, x, v, h)
        print("Complex Step error:       ", r-rr, np.linalg.norm(r-rr))


        points = list(range(-(((i+1) + 1) // 2), 0)) + list(range(1, ((i+1) // 2) + 1))
        fdf_func, _, _ = get_fdf_function(points, i, False)
        r = fdf_func(func_0, x, v, h)
        print("Finite Difference error:  ", r-rr, np.linalg.norm(r-rr))

        print("")

if __name__ == "__main__":
    main()