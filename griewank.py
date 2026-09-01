from basics import Objective_function
import numpy as np
import matplotlib.pyplot as plt 

class Griewank(Objective_function):
    def __init__(self, dim):
        lim_inf = [-600] * dim
        lim_sup = [600] * dim
        super().__init__(dim, lim_inf, lim_sup)

    def function(self, x):
        sum = 0
        product = 1
        for i, xi in enumerate(x, start=1):
            sum = sum + ((xi ** 2) / 4000)
            product = product * np.cos(xi / np.sqrt(i))
        result = sum - product + 1
        return result
    
obj = Griewank(dim=2)
obj.draw3d(name="Griewank")
plt.savefig("griewank_3d.png")
plt.show()

print(obj.function([1, 0]))  
print(obj.function([0, 0]))