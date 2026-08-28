import numpy as np
import matplotlib.pyplot as plt
from dataset import generar_datos

def objective_function(X, y, theta):
    m = len(y)
    J_theta = (np.sum((X @ theta - y) ** 2)) / (2*m)
    return J_theta

def gradient_descent(X, y, theta_inicial, alpha, iteraciones):
    m = len(y)
    costos = []
    theta = theta_inicial
    for i in range(iteraciones):
        gradiente = (1/m) * X.T @ (X @ theta - y)
        theta = theta - alpha * gradiente
        costos.append(objective_function(X, y, theta))
    return theta, costos

if __name__ == "__main__":
    X, y, theta_real = generar_datos()

    costo = objective_function(X, y, theta_real)
    print("Costo con theta_real:", costo)

    theta_prueba = np.zeros(3)
    costo_malo = objective_function(X, y, theta_prueba)
    print("Costo con theta en ceros: ", costo_malo)

    theta_inicial = np.zeros(3)
    theta_final, costos = gradient_descent(X, y, theta_inicial, alpha=0.01, iteraciones=10000)

    print("Theta final:", theta_final)
    print("Theta real:", theta_real)
    print("Último costo:", costos[-1])
    plt.plot(costos)
    plt.xlabel("Iteracion")
    plt.ylabel("J(theta)")
    plt.title("Convergencia de Gradiente Descendente")
    plt.savefig("convergencia_gd.png")
    plt.show()