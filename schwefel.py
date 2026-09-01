from basics import Objective_function
import numpy as np
import matplotlib.pyplot as plt

class Schwefel(Objective_function):
    def __init__(self, dim):
        lim_inf = [-500] * dim
        lim_sup = [500] * dim
        super().__init__(dim, lim_inf, lim_sup)

    def function(self, x):
        d = len(x)
        sum = 0
        for xi in x:
            sum = sum + (xi * np.sin(np.sqrt(np.abs(xi))))
        result = (418.9829 * d) - sum
        return result

obj = Schwefel(dim=2)
obj.draw3d(name="Schwefel")
plt.savefig("schwefel_3d.png")
plt.show()

print(obj.function([1, 1]))  
print(obj.function([0, 0]))
print(obj.function([420.9687, 420.9687]))