# The complex-step derivative approximation
This is a short library that implements the complex-step derivative approximation algorithm for the computation of the N-derivative of an N-dimension function.

This repository also includes the implementation of the multicomplex-step approximation, which prevents cancellation errors on high derivative orders, and an implementation of a variation of the Newton Krylov method present in the Scipy library using this method for the jacobian-vector product approximation instead of the more common finite differences one.

## Examples
Inside the folders, you can find dedicated example scripts.

Calculating the directional derivative at a point:
```python
import numpy as np
from derivative_approximations import get_csa_function, get_fdf_function

cpoints = [1] # points to use for the complex-step approximation
fpoints = [0,1] # points to use for the finite differences approximation

csa_func, _, _ = get_csa_function(cpoints) # complex-step approximation method
fdf_func, _, _ = get_fdf_function(fpoints) # finite differences approximation method

def func(x):
    return np.sin(x)+np.cos(x)

x = np.array([1., 2.]) # Point of interest
v = np.array([1., 1.]) # Direction
h = 1e-8 # Step size

result_csa = csa_func(func, x, v, h)
result_fdf = fdf_func(func, x, v, h)
```

Calculating the 3rd order directional derivative at a point:
```python
import numpy as np
from derivative_approximations import get_csa_function, get_fdf_function
from multicomplex import Multicomplex

cpoints = [1, 2, 3] # points to use for the complex-step approximation
fpoints = [-2,-1,1,2] # points to use for the finite differences approximation

csa_func, _, _ = get_csa_function(cpoints, 3) # complex-step approximation method
fdf_func, _, _ = get_fdf_function(fpoints, 3) # finite differences approximation method

def func(x):
    return np.sin(x)

x = np.array([1., 2., 3.]) # Point of interest
v = np.array([1., 1., 1.]) # Direction
h = 1e-8 # Step size

result_csa  = csa_func(func, x, v, h)
result_fdf  = fdf_func(func, x, v, h)
resutl_mcsa = Multicomplex.complex_step(func, x, v, h, 3)
```

Calculating the root of a function using the Newton Krylov method:
```python
import numpy as np
from scipy.optimize import newton_krylov
from newton_krylov_csa import newton_krylov_csa

# We ensure that there exists at least one root
def _func(x):
    return np.sin(x) * np.cos(x)

xr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
fr = _func(xr)
n = len(xr)

def func(x):
    return _func(x) - fr


x0 = np.zeros(n) # Inital guess
h = 1e-12 # Step size

solution_scipy_method = newton_krylov(func, x0, rdiff=h)
solution_repo_method  = newton_krylov_csa(func, x0, rdiff=h)
```

## Dependencies

* [NumPy](https://numpy.org/)
* [SciPy](https://scipy.org/) (Only for the Newton Krylov method)
