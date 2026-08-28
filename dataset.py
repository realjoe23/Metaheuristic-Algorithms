import numpy as np

def generar_datos(m=100, n=2, seed=42):
    np.random.seed(seed)
    X_raw = np.random.rand(m, n) * 10
    X = np.hstack([np.ones((m, 1)), X_raw])
    theta_real = np.array([5, 2, -3])
    ruido = np.random.randn(m) * 1.0
    y = X @ theta_real + ruido

    return X, y, theta_real

