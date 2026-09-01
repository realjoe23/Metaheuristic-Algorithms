from basics import Objective_function
import numpy as np
import matplotlib.pyplot as plt

class Trid(Objective_function):
    def __init__(self, dim):
        lim_inf = [-(dim ** 2)] * dim
        lim_sup = [(dim ** 2)] * dim
        super().__init__(dim, lim_inf, lim_sup)

    def function(self, x):
        n = len(x)
        sum1 = 0
        for xi in x:
            sum1 = sum1 + ((xi - 1) ** 2)
        sum2 = 0
        for i in range(1, n):
            xi = x[i]
            x_ant = x[i - 1]
            sum2 = sum2 + (xi * x_ant)
        result = sum1 - sum2
        return result

obj = Trid(dim=2)
obj.draw3d(name="Trid")
plt.savefig("trid_3d.png")
plt.show()

print(obj.function([2, 2]))  
print(obj.function([0, 0]))