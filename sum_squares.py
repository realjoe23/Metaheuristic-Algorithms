from basics import Objective_function
import numpy as np
import matplotlib.pyplot as plt

class Sum_squares(Objective_function):
    def __init__(self, dim):
        lim_inf = [-10] * dim
        lim_sup = [10] * dim
        super().__init__(dim, lim_inf, lim_sup)

    def function(self, x):
        sum = 0
        for i, xi in enumerate(x, start=1):
            sum = sum + i * (xi ** 2)
        return sum

obj = Sum_squares(dim=2)
obj.draw3d(name="Sum of Squares")
plt.savefig("sumofsquares_3d.png")
plt.show()

print(obj.function([1, 1]))  
print(obj.function([0, 0]))