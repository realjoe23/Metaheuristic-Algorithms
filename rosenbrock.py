from basics import Objective_function
import numpy as np
import matplotlib.pyplot as plt

class Rosenbrock(Objective_function):
    def __init__(self, dim):
        lim_inf = [-5] * dim
        lim_sup = [10] * dim
        super().__init__(dim, lim_inf, lim_sup)

    def function(self, x):
        n = len(x)
        sum = 0
        for i in range(n-1):
            xi = x[i]
            x_sig = x[i + 1]
            term = (100 * (x_sig - (xi ** 2)) ** 2) + ((xi - 1) ** 2)
            sum = sum + term
        return sum

obj = Rosenbrock(dim=2)
obj.draw3d(name="Rosenbrock")
plt.savefig("rosenbrock_3d.png")
plt.show()

print(obj.function([1, 1]))  
print(obj.function([0, 0]))