from basics import Objective_function
import numpy as np
import matplotlib.pyplot as plt

class Sum_differentpowers(Objective_function):
    def __init__(self, dim):
        lim_inf = [-1] * dim
        lim_sup = [1] * dim
        super().__init__(dim, lim_inf, lim_sup)

    def function(self, x):
        sum = 0
        for i, xi in enumerate(x, start=1):
            sum = sum + abs(xi) ** (i + 1)
        return sum

obj = Sum_differentpowers(dim=2)
obj.draw3d(name="Sum of different powers")
plt.savefig("sumofdifferentpowers_3d.png")
plt.show()

print(obj.function([1, 1]))  
print(obj.function([0, 0]))