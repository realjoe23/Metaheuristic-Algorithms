from basics import Objective_function
import numpy as np
import matplotlib.pyplot as plt

class Bochachevsky(Objective_function):
    def __init__(self, dim):
        lim_inf = [-100] * dim
        lim_sup = [100] * dim
        super().__init__(dim, lim_inf, lim_sup)

    def function(self, x):
        n = len(x)
        sum = 0
        for i in range(n-1):
            xi = x[i]
            x_sig = x[i + 1]
            term = (xi ** 2) + (2*(x_sig ** 2)) - 0.3*np.cos(3*np.pi*xi) - 0.4*np.cos(4*np.pi*x_sig) + 0.7
            sum = sum + term
        return sum

obj = Bochachevsky(dim=2)
obj.draw3d(name="Bochachevsky")
plt.savefig("bochachevsky_3d.png")
plt.show()

print(obj.function([1, 1]))  
print(obj.function([0, 0]))