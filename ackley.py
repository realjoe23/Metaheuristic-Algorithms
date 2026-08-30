from basics import Objective_function
import numpy as np
import matplotlib.pyplot as plt

class Ackley(Objective_function):
    def __init__(self, dim):
        lim_inf = [-32.768] * dim
        lim_sup = [32.768] * dim
        super().__init__(dim, lim_inf, lim_sup)

    def function(self, x):
        n = len(x)
        a = 20
        b = 0.2
        c = 2*np.pi
        sum1 = 0
        sum2 = 0

        for xi in x:
            sum1 = sum1 + (xi ** 2)
            sum2 = sum2 + np.cos(c * xi)

        part1 = np.sqrt(sum1 / n)
        part2 = sum2 / n

        result = -a * np.exp(-b * part1) - np.exp(part2) + a + np.e
        return result


obj = Ackley(dim=2)
obj.draw3d(name="Ackley")
plt.savefig("ackley_3d.png")
plt.show()

print(obj.function([0,0]))