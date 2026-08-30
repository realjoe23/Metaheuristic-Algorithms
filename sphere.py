from basics import Objective_function
import numpy as np
import matplotlib.pyplot as plt

class Sphere(Objective_function):
    def __init__(self, dim):
        lim_inf = [-5.12] * dim
        lim_sup = [5.12] * dim
        super().__init__(dim, lim_inf, lim_sup)

    def function(self, x):
        sum = 0
        for xi in x:
            sum = sum + (xi ** 2)
        return sum

obj = Sphere(dim=2)
obj.draw3d(name="Sphere")
plt.savefig("sphere_3d.png")
plt.show()
