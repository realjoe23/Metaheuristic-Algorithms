
import numpy as np
import matplotlib.pyplot as plt
from basics import Objective_function, Animation

def exhaustive_search(obj, n_points, batch_size=None):
    """
    Búsqueda exhaustiva sobre una malla uniforme del dominio de obj.

    Parameters
    ----------
    obj : Objective_function
        Instancia con obj.d, obj.lb, obj.ub y obj.eval(X) definidos.
    n_points : int
        Número de puntos por eje de la malla.
    batch_size : int, opcional
        Cuántos puntos de la malla se agrupan por cuadro de animación.
        Si es None, se calcula para tener aproximadamente 50 cuadros.

    Returns
    -------
    best_x : np.array de forma (d,)
    best_f : float
    lotes : list de (X_lote, ig_local)
        Listo para pasarle directo a Animation.
    """
    ejes = [np.linspace(obj.lb[i], obj.ub[i], n_points) for i in range(obj.d)]

    mesh = np.meshgrid(*ejes)
    X = np.vstack([m.ravel() for m in mesh])

    valores = obj.eval(X)

    best_idx = np.argmin(valores)
    best_x = X[:, best_idx]
    best_f = valores[best_idx]

    if batch_size is None:
        batch_size = max(1, X.shape[1] // 50)

    lotes = []
    for start in range(0, X.shape[1], batch_size):
        end = start + batch_size
        X_lote = X[:, start:end]
        f_lote = valores[start:end]
        ig_local = np.argmin(f_lote)
        lotes.append((X_lote, ig_local))

    return best_x, best_f, lotes


if __name__ == "__main__":
    from rastringin import Rastringin

    obj = Rastringin(dim=2)
    n_points = 40

    best_x, best_f, lotes = exhaustive_search(obj, n_points=n_points)

    print("Mejor x:", best_x)
    print("Mejor f(x):", best_f)
    print("Número de lotes (cuadros de animación):", len(lotes))

    anim = Animation(animation_type='3d', name="Exhaustive Search - Rastringin")
    X0, ig0 = lotes[0]
    anim.initialize_animation(obj, X0, ig0)
    for X_lote, ig in lotes[1:]:
        anim.update_animation(obj, X_lote, ig)

    plt.show()