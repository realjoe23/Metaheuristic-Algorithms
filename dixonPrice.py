from basics import Objective_function
import numpy as np
import matplotlib.pyplot as plt

class Dixon_Price(Objective_function):
    def __init__(self, dim):
        lim_inf = [-10] * dim
        lim_sup = [10] * dim
        super().__init__(dim, lim_inf, lim_sup)

    def function(self, x):
        n = len(x)
        first_term = ((x[0] - 1) ** 2)
        sum = 0
        for i in range(1, n):
            xi = x[i]
            x_ant = x[i-1]
            index_formula = i + 1
            term = index_formula * (((2*(xi ** 2)) - x_ant) ** 2)
            sum = sum + term
        result = first_term + sum
        return result

obj = Dixon_Price(dim=2)
obj.draw3d(name="Dixon Price")
plt.savefig("dixon_price_3d.png")
plt.show()

print(obj.function([1, 1]))  
print(obj.function([0, 0]))