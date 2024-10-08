import numpy as np
import scipy.sparse.linalg
from scipy.linalg import norm
import scipy.optimize._nonlin
from scipy.optimize._nonlin import Jacobian, _nonlin_wrapper

_scipy_version = scipy.__version__
_scipy_version_parts = _scipy_version.split('.')
scipy_version = int(_scipy_version_parts[1])

class KrylovJacobianCSA(Jacobian):

    r"""

    Works the same as KrylovJacobian but uses Complex Step approximation instead of forward difference.
    Only "matvec(self, v)" method is modified.

    Examples
    --------
    The following functions define a system of nonlinear equations

    >>> def fun(x):
    ...     return [x[0] + 0.5 * x[1] - 1.0,
    ...             0.5 * (x[1] - x[0]) ** 2]

    A solution can be obtained as follows.

    >>> from newton_krylov_csa import newton_krylov_csa
    >>> sol = newton_krylov_csa(fun, [0, 0])
    >>> sol
    array([0.66731771, 0.66536458])

    """

    def __init__(self, rdiff=None, method='lgmres', inner_maxiter=20,
                 inner_M=None, outer_k=10, **kw):
        self.preconditioner = inner_M
        self.rdiff = rdiff

        # Note that this retrieves one of the named functions, or otherwise
        # uses `method` as is (i.e., for a user-provided callable).
        self.method = dict(
            bicgstab=scipy.sparse.linalg.bicgstab,
            gmres=scipy.sparse.linalg.gmres,
            lgmres=scipy.sparse.linalg.lgmres,
            cgs=scipy.sparse.linalg.cgs,
            minres=scipy.sparse.linalg.minres,
            tfqmr=scipy.sparse.linalg.tfqmr,
            ).get(method, method)

        self.method_kw = dict(maxiter=inner_maxiter, M=self.preconditioner)

        if self.method is scipy.sparse.linalg.gmres:
            # Replace GMRES's outer iteration with Newton steps
            self.method_kw['restart'] = inner_maxiter
            self.method_kw['maxiter'] = 1
            self.method_kw.setdefault('atol', 0)
        elif self.method in (scipy.sparse.linalg.gcrotmk,
                             scipy.sparse.linalg.bicgstab,
                             scipy.sparse.linalg.cgs):
            self.method_kw.setdefault('atol', 0)
        elif self.method is scipy.sparse.linalg.lgmres:
            self.method_kw['outer_k'] = outer_k
            # Replace LGMRES's outer iteration with Newton steps
            self.method_kw['maxiter'] = 1
            # Carry LGMRES's `outer_v` vectors across nonlinear iterations
            self.method_kw.setdefault('outer_v', [])
            self.method_kw.setdefault('prepend_outer_v', True)
            # But don't carry the corresponding Jacobian*v products, in case
            # the Jacobian changes a lot in the nonlinear step
            #
            # XXX: some trust-region inspired ideas might be more efficient...
            #      See e.g., Brown & Saad. But needs to be implemented separately
            #      since it's not an inexact Newton method.
            self.method_kw.setdefault('store_outer_Av', False)
            self.method_kw.setdefault('atol', 0)

        for key, value in kw.items():
            if not key.startswith('inner_'):
                raise ValueError("Unknown parameter %s" % key)
            self.method_kw[key[6:]] = value

    def matvec(self, v): # Changed to complex step.
        nv = norm(v)
        if nv == 0:
            return 0*v
        sc = self.rdiff / nv
        r = np.imag(self.func(self.x0 + sc * v * 1j)) / sc
        if not np.all(np.isfinite(r)) and np.all(np.isfinite(v)):
            raise ValueError('Function returned non-finite results')
        return r

    def solve(self, rhs, tol=0):
        if scipy_version >= 14:
            if 'rtol' in self.method_kw:
                sol, info = self.method(self.op, rhs, **self.method_kw)
            else:
                sol, info = self.method(self.op, rhs, rtol=tol, **self.method_kw)
            return sol
        else:
            if 'tol' in self.method_kw:
                sol, info = self.method(self.op, rhs, **self.method_kw)
            else:
                sol, info = self.method(self.op, rhs, tol=tol, **self.method_kw)
            return sol

    def update(self, x, f):
        self.x0 = x
        self.f0 = f

        # Update also the preconditioner, if possible
        if self.preconditioner is not None:
            if hasattr(self.preconditioner, 'update'):
                self.preconditioner.update(x, f)

    def setup(self, x, f, func):
        Jacobian.setup(self, x, f, func)
        self.x0 = x
        self.f0 = f
        self.op = scipy.sparse.linalg.aslinearoperator(self)

        if self.rdiff is None:
            self.rdiff = np.finfo(x.dtype).eps ** (1./2)

        # Setup also the preconditioner, if possible
        if self.preconditioner is not None:
            if hasattr(self.preconditioner, 'setup'):
                self.preconditioner.setup(x, f, func)

scipy.optimize._nonlin.KrylovJacobianCSA = KrylovJacobianCSA
                
newton_krylov_csa = _nonlin_wrapper('newton_krylov_csa', scipy.optimize._nonlin.KrylovJacobianCSA)
