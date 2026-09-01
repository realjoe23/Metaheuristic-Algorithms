from basics import Objective_function
import numpy as np
import matplotlib.pyplot as plt

class Rotated_Hyperellipsoid(Objective_function):
    def __init__(self, dim):
        lim_inf = [-65.536] * dim
        lim_sup = [65.536] * dim
        super().__init__(dim, lim_inf, lim_sup)

    def function(self, x):
        n = len(x)
        sum_ext = 0
        for i in range(0, n):
            sum_int = 0
            for j in range(0, i + 1):
                sum_int = sum_int + x[j]
            sum_ext = sum_ext + (sum_int ** 2)
        return sum_ext

obj = Rotated_Hyperellipsoid(dim=2)
obj.draw3d(name="Rotated Hyperellipsoid")
plt.savefig("rotated_hyperellipsoid_3d.png")
plt.show()

print(obj.function([1, 1]))  
print(obj.function([0, 0]))
print(obj.function([1, 2]))
