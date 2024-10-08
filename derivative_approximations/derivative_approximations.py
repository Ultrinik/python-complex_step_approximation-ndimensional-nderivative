import numpy as np
from scipy.special import factorial

def get_last_non_zero(a):
    for i in range(len(a) - 1, -1, -1):
        if a[i] != 0:
            return a[i], i
    return None, None

def get_csa_function(points: list[float], derivate: int = 1, print_mode: bool = False, dtype: type = np.complex128) -> tuple[callable, tuple, list]:
    """
    Returns a Complex Step Approximation function constructed from the given points.

    Parameters
    ----------
    points : list
        List of points to use for the approximation.
    derivate : int, optional
        The order of the derivative to calculate. Default is `1`.
    print_mode : bool, optional
        Whether to enable debug prints (True for debugging). Default is `False`.
    dtype : type, optional
        Complex number data type to use, typically either `np.complex128` or `np.complex64`. Default is `np.complex128`.

    Returns
    -------
    callable
        The resulting function that takes the arguments (func, x, v, h).
    tuple
        A 2-tuple containing the convergence order of the approximation.
    list
        List of coefficients used in the approximation, in the same order as the input `points` list.
    """

    # Remove point 0 if derivate is even, it will always be 0
    if derivate%2==1 and 0 in points: points.pop(points.index(0))
    # Decide if method needs real or imaginary component
    component_func = np.imag if derivate%2==1 else np.real

    # Get coefficients of Taylor Expansion
    def get_coefficients(points):
        coefficients = []
        for point in points:
            current_coefficients = np.zeros(2 * len(points), dtype=dtype)
            for i in range(0, 2 * len(points)):
                power = dtype(1j * point) ** i
                coefficient = power / factorial(i)
                current_coefficients[i] = coefficient
            coefficients.append(current_coefficients)
        return component_func(np.array(coefficients).T)
    coefficients = get_coefficients(points)

    # Get coefficients of Taylor Expansion with one more point, so it can be used to calculate convergence
    points.append(max(np.abs(points))+1)
    extended_coefficients_og = get_coefficients(points)
    extended_non_zero_rows = ~np.all(extended_coefficients_og == 0, axis=1)
    extended_coefficients_og_filtered = extended_coefficients_og[extended_non_zero_rows]
    extended_coefficients_filtered = extended_coefficients_og_filtered.copy()[:, :-1]

    # Solve system
    A = coefficients
    b = np.zeros(A.shape[0])
    b[derivate] = 1

    non_zero_rows = ~np.all(A == 0, axis=1)
    A_filtered = A[non_zero_rows]
    b_filtered = b[non_zero_rows]

    ## Debug
    if print_mode:
        print("Matrix A_filtered:")
        print(A_filtered)
        print("Array b_filtered:")
        print(b_filtered)

    solution = None
    for i in range(5):
        try:
            solution = np.linalg.solve(A_filtered, b_filtered)
            break
        except np.linalg.LinAlgError as e:
            A_filtered = A_filtered[:-1, :-1]
            b_filtered = b_filtered[:-1]

            if print_mode:
                print("Error: A complex-step method can not be generated with the chosen points.")
                if i < 4: print("Adjusting Points")
                
                print(f"({i+1}) Matrix A:")
                print(A_filtered)
                print(f"({i+1}) Array b:")
                print(b_filtered)
        except Exception as e:
            print("An unexpected error occurred:", e)
            return

    # Convergence
    extended_coefficients_filtered_cutted = extended_coefficients_filtered[:, :len(solution)]
    extended_b = extended_coefficients_filtered_cutted @ solution

    extended_b_non_zero_values = extended_b[extended_b != 0]
    second_non_zero_value = extended_b_non_zero_values[1]
    second_non_zero_index = np.where(extended_b == second_non_zero_value)[0][0]

    o_factor = second_non_zero_value
    o_factor_exp = 2*second_non_zero_index

    final_solution = solution.copy()

    # Debug
    if print_mode:
        print("Solution X:")
        print(final_solution)
        print("A @ Solution:")
        print(A_filtered @ final_solution)
        print("ext_A @ Solution:")
        print(extended_b)
        print("Order:")
        print(f"O({abs(o_factor)} h^{o_factor_exp})")

    # Function
    points = np.array(points).astype(dtype)
    final_solution = np.array(final_solution).astype(dtype)
    def csa_function(F: callable, x: float | np.ndarray, v: float | np.ndarray, h: float = 1e-8) -> float | np.ndarray:
        """
        Computes the Complex Step Approximation of the function F at a point x.

        Parameters
        ----------
        F : callable
            A function that takes a single input (float or array).
        x : float | np.ndarray
            The point(s) at which to evaluate the approximation.
        v : float | np.ndarray
            The perturbation values for the approximation, must be of the same type as `x`.
        h : float, optional
            The step size for the approximation. Default is `1e-8`.

        Returns
        -------
        float | np.ndarray
            The approximated value(s) of the function F at the specified point x.
        """
        total = np.zeros_like(x, dtype=dtype)
        for i in range(len(final_solution)):
            point = points[i]
            final_c = final_solution[i]

            step = v*h*1j
            arguments = dtype(x + step*point)
            
            total += final_c * F(arguments)

        return component_func(total)/h**derivate
    
    return csa_function, (o_factor, o_factor_exp), final_solution

def get_fdf_function(points: list[float], derivate: int = 1, print_mode: bool = False, dtype: type = np.float64) -> tuple[callable, tuple, list]:
    """
    Returns a Finite Difference Approximation function constructed from the given points.

    Parameters
    ----------
    points : list
        List of points to use for the approximation.
    derivate : int, optional
        The order of the derivative to calculate. Default is `1`.
    print_mode : bool, optional
        Whether to enable debug prints (True for debugging). Default is `False`.
    dtype : type, optional
        Data type that the function will use, typically `np.float64`. Default is `np.float64`.

    Returns
    -------
    callable
        The resulting function that takes the arguments (func, x, v, h).
    tuple
        A 2-tuple containing the convergence order of the approximation.
    list
        List of coefficients used in the approximation, in the same order as the input `points` list.
    """

    # Get coefficients of Taylor Expansion
    def get_coefficients(points):
        coefficients = []
        for point in points:
            current_coefficients = np.zeros(len(points), dtype=dtype)
            for i in range(0, len(points)):
                power = point ** i
                coefficient = dtype(power / factorial(i))
                current_coefficients[i] = coefficient
            coefficients.append(current_coefficients)
        return np.array(coefficients).T
    coefficients = get_coefficients(points)

    # Get coefficients of Taylor Expansion with one more point, so it can be used to calculate convergence
    points.append(max(np.abs(points)) + 1)
    extended_coefficients_og = get_coefficients(points)
    extended_coefficients = extended_coefficients_og.copy()[:, :-1]

    # Solve system
    A = coefficients
    b = np.zeros(A.shape[0])
    b[derivate] = 1
    solution = np.linalg.solve(A, b)

    # Convergence
    extended_b = extended_coefficients @ solution
    o_factor, o_factor_exp = get_last_non_zero(extended_b)
    o_factor_exp -= 1

    final_solution = solution.copy()

    # Debug
    if print_mode:
        print("Matrix A:")
        print(A)
        print("Array b:")
        print(b)
        print("Solution X:")
        print(final_solution)
        print("A @ Solution:")
        print(coefficients @ final_solution)
        print("ext_A @ Solution:")
        print(extended_b)
        print("Order:")
        print(f"O({abs(o_factor)} h^{o_factor_exp})")

    # Function
    def fdf_function(F: callable, x: float | np.ndarray, v: float | np.ndarray, h: float = 1e-8) -> float | np.ndarray:
        """
        Computes the Finite Difference approximation of the function F at a point x.

        Parameters
        ----------
        F : callable
            A function that takes a single input (float or array).
        x : float | np.ndarray
            The point(s) at which to evaluate the approximation.
        v : float | np.ndarray
            The perturbation values for the approximation, must be of the same type as `x`.
        h : float, optional
            The step size for the approximation. Default is `1e-8`.

        Returns
        -------
        float | np.ndarray
            The approximated value(s) of the function F at the specified point x.
        """
        total = 0 * x
        for i in range(len(points) - 1):
            point = points[i]
            final_c = final_solution[i]
            step = v * h
            total += final_c * F(x + step * point)
        return total / h ** derivate

    return fdf_function, (o_factor, o_factor_exp), final_solution