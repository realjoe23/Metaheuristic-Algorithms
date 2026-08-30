from basics import Objective_function
import numpy as np
import matplotlib.pyplot as plt

class Rastringin(Objective_function):
    def __init__(self, dim):
        lim_inf = [-5.12] * dim
        lim_sup = [5.12] * dim
        super().__init__(dim, lim_inf, lim_sup)

    def function(self, x):
        n = len(x)
        sum = 0
        for xi in x:
            sum = sum + ((xi ** 2) - (10 * np.cos(2*np.pi*xi)))
        result = (10*n) + sum
        return result

obj = Rastringin(dim=2)
obj.draw3d(name="Rastringin")
plt.savefig("rastringin_3d.png")
plt.show()

print(obj.function([0, 0]))  
print(obj.function([1, 1]))  
