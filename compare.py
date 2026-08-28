from dataset import generar_datos
from gradient_descent import gradient_descent, objective_function
from pseudoinverse import pseudoinversa
import numpy as np
import matplotlib.pyplot as plt

X, y, theta_real = generar_datos()

theta_gd, costos = gradient_descent(X, y, np.zeros(3), alpha=0.01, iteraciones=10000)
theta_pinv = pseudoinversa(X, y)
costo_pinv = objective_function(X, y, theta_pinv)

plt.plot(costos, label="Gradiente Descendente")
plt.axhline(y=costo_pinv, color='r', linestyle='--', label="Pseudoinversa (óptimo)")
plt.xlabel("Iteracion")
plt.ylabel("J(theta)")
plt.title("Convergencia de GD vs. Pseudoinversa")
plt.legend()
plt.savefig("comparacion_gd_pinv.png")
plt.show()