import numpy as np
import matplotlib.pyplot as plt
from basics import Objective_function, Animation

def random_search(obj, n_samples, batch_size=None, seed=None):

    if seed is not None:
        np.random.seed(seed)

    lb = np.array(obj.lb).reshape(-1, 1)
    ub = np.array(obj.ub).reshape(-1, 1)
    X = lb + np.random.rand(obj.d, n_samples) * (ub - lb)

    valores = obj.eval(X)

    best_idx = np.argmin(valores)
    best_x = X[:, best_idx]
    best_f = valores[best_idx]

    if batch_size is None:
        batch_size = max(1, n_samples // 50)

    lotes = []
    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        X_lote = X[:, start:end]
        f_lote = valores[start:end]
        ig_local = np.argmin(f_lote)
        lotes.append((X_lote, ig_local))

    return best_x, best_f, lotes

if __name__ == "__main__":
    from sphere import Sphere  

    obj = Sphere(dim=2)
    n_samples = 1600  

    best_x, best_f, lotes = random_search(obj, n_samples=n_samples, seed=42)

    print("Mejor x:", best_x)
    print("Mejor f(x):", best_f)
    print("Número de lotes (cuadros de animación):", len(lotes))

    anim = Animation(animation_type='3d', name="Random Search - Sphere")
    X0, ig0 = lotes[0]
    anim.initialize_animation(obj, X0, ig0)
    for X_lote, ig in lotes[1:]:
        anim.update_animation(obj, X_lote, ig)

    plt.show()