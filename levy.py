from basics import Objective_function
import numpy as np
import matplotlib.pyplot as plt

class Levy(Objective_function):
    def __init__(self, dim):
        lim_inf = [-10] * dim
        lim_sup = [10] * dim
        super().__init__(dim, lim_inf, lim_sup)

    def function(self, x):
        n = len(x)
        w = [0] * n
        for i in range(n):
            w[i] = 1 + ((x[i] - 1) / 4)
        term1 = np.sin(np.pi * w[0]) ** 2
        sum_mid = 0
        for i in range(n-1):
            wi = w[i]
            parte = ((wi - 1) ** 2 ) * (1 + 10 * (np.sin(np.pi * wi + 1) ** 2))
            sum_mid = sum_mid + parte
        w_last = w[n-1]
        term3 = ((w_last - 1) ** 2) * (1 + (np.sin(2*np.pi*w_last) ** 2)) 
        result = term1 + sum_mid + term3
        return result

obj = Levy(dim=2)
obj.draw3d(name="Levy")
plt.savefig("levy_3d.png")
plt.show()

print(obj.function([1, 1]))  
print(obj.function([0, 0]))