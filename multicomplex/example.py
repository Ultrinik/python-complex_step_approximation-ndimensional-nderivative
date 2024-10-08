import numpy as np
from multicomplex import Multicomplex

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

      print("")

if __name__ == "__main__":
   main()