from basics import Objective_function
import numpy as np
import matplotlib.pyplot as plt

class Zakharov(Objective_function):
    def __init__(self, dim):
        lim_inf = [-5] * dim
        lim_sup = [10] *dim
        super().__init__(dim, lim_inf, lim_sup)

    def function(self, x):
        sumA = 0
        sumB = 0
        for i, xi in enumerate(x, start=1):
            sumA = sumA + (xi ** 2)
            sumB = sumB + (0.5 * i * xi)
        result = sumA + (sumB ** 2) + (sumB ** 4)
        return result

obj = Zakharov(dim=2)
obj.draw3d(name="Zakharov")
plt.savefig("zakharov_3d.png")
plt.show()

print(obj.function([1, 0]))  
print(obj.function([0, 0]))