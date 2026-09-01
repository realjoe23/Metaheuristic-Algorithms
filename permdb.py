from basics import Objective_function
import numpy as np
import matplotlib.pyplot as plt

class Permdb(Objective_function):
    def __init__(self, dim):
        lim_inf = [-dim] * dim
        lim_sup = [dim] * dim
        super().__init__(dim, lim_inf, lim_sup)

    def function(self, x): 
        d = len(x)
        beta = 0.5
        sum_ext = 0
        for i in range(1, d + 1):
            sum_int = 0
            for j in range(1, d + 1):
                xj = x[j - 1]
                term = ((j ** i) + beta) * (((xj / j) ** i) - 1)
                sum_int = sum_int + term
            sum_ext = sum_ext + (sum_int ** 2)
        return sum_ext

obj = Permdb(dim=2)
obj.draw3d(name="Permdb")
plt.savefig("permdb_3d.png")
plt.show()

print(obj.function([1, 2]))
print(obj.function([0, 0]))