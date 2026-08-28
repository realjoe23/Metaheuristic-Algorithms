import numpy as np
from dataset import generar_datos
from gradient_descent import objective_function

def pseudoinversa(X, y):
    theta = np.linalg.pinv(X) @ y
    return theta

if __name__ == "__main__":
    X, y, theta_real = generar_datos()

    theta_pinv = pseudoinversa(X, y)
    print("Theta pseudoinversa:", theta_pinv)
    print("Theta real:", theta_real)

    costo_pinv = objective_function(X, y, theta_pinv)
    print("Costo con pseudoinversa:", costo_pinv)