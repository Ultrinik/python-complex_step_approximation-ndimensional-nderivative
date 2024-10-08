import numpy as np
import math

def binary_list_to_int(binary_list):
    binary_string = ''.join(map(str, binary_list))
    decimal_number = int(binary_string, 2)
    return decimal_number

def getImag(X, flags = [1]):
  """
  Returns a real number corresponding with the imputed imaginary component.

  Parameters
  ----------
  X : Multicomplex
    Multicomplex number.
  flags : Int|List
    Pseudo bit mask representing the desired product of imaginary components.

  Returns
  -------
  Float
    The coefficient that corresponds to the product of the imaginary components.
  """
  nFlag = 0
  if isinstance(flags, int):
    nFlag = flags
  else:
    nFlag = binary_list_to_int(flags)

  return X[nFlag]

# CLASS
_tiny_name = 'tiny' if np.__version__ < '1.22' else 'smallest_normal'
_TINY = getattr(np.finfo(float), _tiny_name)

Mdtype=np.float64

class Multicomplex(object):

  def __init__(self, nums, z2=None):

    if z2 is not None:  # Two parameters input
      self._initialize_two_parameters(nums, z2)
    else:  # One parameter input
      nums = self._convert_to_list(nums)
      self._initialize_one_parameter(nums)

  def _initialize_two_parameters(self, z1, z2):
    # Rank
    if isinstance(z1, (int, float)):
      self._rank = 1
    elif isinstance(z1, complex):
      self._rank = 2
    else:
      self._rank = z1._rank + 1
    # Components
    self.z1 = z1
    self.z2 = z2

  def _convert_to_list(self, nums):
    if isinstance(nums, (np.ndarray, list)):
      return nums
    elif isinstance(nums, complex):
      return [nums.real, nums.imag]
    else:
      return [nums]

  def _initialize_one_parameter(self, v):
    l = len(v)
    self._rank = int(np.log2(l))

    if l == 1:
        self.z1 = v[0]
        self.z2 = 0
    elif l == 2:
        self.z1 = v[0]
        self.z2 = v[1]
    elif l == 4:
        self.z1 = np.complex128(v[0] + v[1] * 1j)
        self.z2 = np.complex128(v[2] + v[3] * 1j)
    else:
        midpoint = l // 2
        self.z1 = Multicomplex(v[:midpoint])
        self.z2 = Multicomplex(v[midpoint:])

  @property
  def length(self):
    return 2**self.rank

  def __len__(self):
    return self.length

  @property
  def rank(self):
    return self._rank

  @property
  def total(self):
    return sum(self.asarray)

  def mod_c(self):
    """Complex modulus"""
    r11, r22 = self.z1 * self.z1, self.z2 * self.z2
    r = np.sqrt(r11 + r22)
    return r

  def norm(self):
    return np.linalg.norm(self.asarray)

  @property
  def real(self):
    return self.z1.real

  def imag(self, flags):
    return getImag(self.asarray, flags)

  @property
  def asarray(self):
    z1, z2 = self.z1, self.z2
    if isinstance(z1, (int, float)):
      return np.array([z1, z2], dtype=Mdtype)
    elif isinstance(z1, (complex)):
      return np.array([z1.real, z1.imag, z2.real, z2.imag], dtype=Mdtype)
    return np.block([z1.asarray, z2.asarray])

  @staticmethod
  def _coerce(other):
    if isinstance(other, Multicomplex):
      return other
    return Multicomplex(other)

  def __eq__(self, other):
    return np.array_equal(self.asarray, other.asarray)

  def __setitem__(self, index, value):
    print("This method is WIP")

  def __abs__(self):
    return self.norm()

  def __neg__(self):
    return Multicomplex(-self.z1, -self.z2)

  def __add__(self, other):
    other = self._coerce(other)
    return Multicomplex(self.z1 + other.z1, self.z2 + other.z2)

  def __sub__(self, other):
    other = self._coerce(other)
    return Multicomplex(self.z1 - other.z1, self.z2 - other.z2)

  def __rsub__(self, other):
    return -self.__sub__(other)

  def __div__(self, other):
    result = self * other ** -1
    return result

  def __rdiv__(self, other):
    return self.__div__(other)

  __truediv__ = __div__
  __rtruediv__ = __rdiv__

  def __mul__(self, other):
    other = self._coerce(other)
    return Multicomplex(self.z1 * other.z1 - self.z2 * other.z2,
                      self.z1 * other.z2 + self.z2 * other.z1)

  def __pow__(self, exponent):
    if self.total == 0:
      return self

    return (self.log() * exponent).exp()

  def sqrt(self):
    return self.__pow__(0.5)

  __radd__ = __add__
  __rmul__ = __mul__

  def conjugate(self):
    return Multicomplex(self.z1, -self.z2)

  def flat(self, index):
    return Multicomplex(self.z1.flat[index], self.z2.flat[index])

  def dot(self, other):
    other = self._coerce(other)
    result = self.asarray * other.asarray
    return Multicomplex(result)

  def logaddexp(self, other):
    #TODO?
    other = self._coerce(other)
    return self + np.log1p(np.exp(other - self))

  def logaddexp2(self, other):
    #TODO?
    other = self._coerce(other)
    return self + np.log2(1 + np.exp2(other - self))

  def sin(self):
    z1 = np.cosh(self.z2) * np.sin(self.z1)
    z2 = np.sinh(self.z2) * np.cos(self.z1)
    return Multicomplex(z1, z2)

  def cos(self):
    z1 = np.cosh(self.z2) * np.cos(self.z1)
    z2 = -np.sinh(self.z2) * np.sin(self.z1)
    return Multicomplex(z1, z2)

  def tan(self):
    return self.sin() / self.cos()

  def cot(self):
    return self.cos() / self.sin()

  def sec(self):
    return 1. / self.cos()

  def csc(self):
    return 1. / self.sin()

  def cosh(self):
    z1 = np.cosh(self.z1) * np.cos(self.z2)
    z2 = np.sinh(self.z1) * np.sin(self.z2)
    return Multicomplex(z1, z2)

  def sinh(self):
    z1 = np.sinh(self.z1) * np.cos(self.z2)
    z2 = np.cosh(self.z1) * np.sin(self.z2)
    return Multicomplex(z1, z2)

  def tanh(self):
    return self.sinh() / self.cosh()

  def coth(self):
    return self.cosh() / self.sinh()

  def sech(self):
    return 1. / self.cosh()

  def csch(self):
    return 1. / self.sinh()

  def exp2(self):
    return np.exp(self * np.log(2))

  def log10(self):
    return self.log() / np.log(10)

  def log2(self):
    return self.log() / np.log(2)

  def log1p(self):
    return Multicomplex(np.log1p(self.mod_c()), self.arg_c1p())

  def exp(self):
    expz1 = np.exp(self.z1)
    if np.abs(expz1) < 1e-15:
      return Multicomplex(expz1, expz1)
    return Multicomplex(expz1 * np.cos(self.z2), expz1 * np.sin(self.z2))

  expm1 = exp

  def log(self):
    mod_c = self.mod_c()
    arg = np.arctan(self.z2/(self.z1 + _TINY*2))

    return Multicomplex(np.log(mod_c + _TINY*2), arg)

  def arcsin(self):
    J = Multicomplex([0,0, 1,0])
    return -J * ((J * self + (1 - self ** 2) ** 0.5).log())

  def arccos(self):
    return np.pi / 2 - self.arcsin()

  def arctan(self):
    J = Multicomplex([0,0, 1,0])
    arg1, arg2 = 1 - J * self, 1 + J * self
    tmp = J * (arg1.log() - arg2.log()) * 0.5
    result = Multicomplex(tmp.z1, tmp.z2)
    return result

  def arccosh(self):
    return (self + (self ** 2 - 1) ** 0.5).log()

  def arcsinh(self):
    return (self + (self ** 2 + 1) ** 0.5).log()

  def arctanh(self):
    return 0.5 * (((1 + self) / (1 - self)).log())

  def __str__(self):
    return "Multicomplex: " + np.array2string(self.asarray)

  @staticmethod
  def _arg_c(z1, z2):
    sign = np.where((z1.real == 0) * (z2.real == 0), 0, np.where(0 <= z2.real, 1, -1))
    
    arg = z2 / (z1 + _TINY)
    arg_c = np.arctan(arg) + sign * np.pi * (z1.real <= 0)
    return arg_c

  def arg_c1p(self):
    z1, z2 = 1 + self.z1, self.z2
    return self._arg_c(z1, z2)

  def arg_c(self):
    return self._arg_c(self.z1, self.z2)

  def clip(self, min, max):
    v = self.asarray
    v = np.clip(v, min, max)
    return Multicomplex(v)

  @staticmethod
  def to_multicomplex_arr(x):
    n = len(x)
    new_x = np.array([Multicomplex(0)]*n, dtype=Multicomplex)
    for i in range(n):
      new_x[i] = Multicomplex(x[i])
    return new_x

  @staticmethod
  def complex_step(func: callable, x: float | np.ndarray, v: float | np.ndarray, h: float = 1e-8, derivate: int = 1) -> float | np.ndarray:
    """
    Returns the multicomplex step approximation of an nth-order derivative of a function at point x, in the direction v.

    Parameters
    ----------
    func : callable
        The function whose derivative will be approximated.
    x : float | np.ndarray
        The point of interest where the derivative will be calculated.
    v : float | np.ndarray
        The direction in which the derivative is to be calculated.
    h : float
        Step size for the approximation. Default is 1e-8.
    derivate : int
        The order of the derivative to be calculated. Default is 1.

    Returns
    -------
    float | np.ndarray
        The calculated derivative value.
    """

    input = [0]*(2**derivate)
    for i in range(derivate):
      input[(1<<i)] = 1
    input = Multicomplex(input)

    complex_step = (h * v)
    new_complex_step = np.array([input]*len(complex_step), dtype=Multicomplex)
    for i in range(len(complex_step)):
      new_complex_step[i] *= complex_step[i]

    x = Multicomplex.to_multicomplex_arr(x)
    new_x = x + new_complex_step

    flags = 2**(derivate)-1

    result = func(new_x)
    new_result = np.array([0.0]*len(result), dtype=Mdtype)
    for i in range(len(result)):
      new_result[i] = result[i].imag(flags)
    return  new_result / (h**derivate)