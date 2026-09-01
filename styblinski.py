from basics import Objective_function
import numpy as np
import matplotlib.pyplot as plt

class Styblinski(Objective_function):
    def __init__(self, dim):
        lim_inf = [-5] * dim
        lim_sup = [5] * dim
        super().__init__(dim, lim_inf, lim_sup)

    def function(self, x):
        sum = 0
        for xi in x:
            sum = sum + ((xi ** 4) - (16 * (xi ** 2)) + (5 * xi))
        result = 0.5 * sum
        return result

obj = Styblinski(dim=2)
obj.draw3d(name="Styblinski")
plt.savefig("styblinski_3d.png")
plt.show()

print(obj.function([1, 1]))  
print(obj.function([0, 0]))