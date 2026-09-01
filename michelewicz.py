from basics import Objective_function
import numpy as np
import matplotlib.pyplot as plt 

class Michelewicz(Objective_function):
    def __init__(self, dim):
        lim_inf = [0] * dim
        lim_sup = [np.pi] * dim
        super().__init__(dim, lim_inf, lim_sup)

    def function(self, x):
        m = 10
        sum = 0
        for i, xi in enumerate(x, start=1):
            parte_int = np.sin(i * ((xi ** 2) / np.pi))
            term = np.sin(xi) * (parte_int ** (2*m))
            sum = sum + term
        result = -sum
        return result

obj = Michelewicz(dim=2)
obj.draw3d(name="Michelewicz")
plt.savefig("michelewicz_3d.png")
plt.show()

print(obj.function([0, 0]))
print(obj.function([1, 1]))