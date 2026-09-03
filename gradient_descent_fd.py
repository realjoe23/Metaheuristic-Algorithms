import numpy as np
import matplotlib.pyplot as plt
from basics import Objective_function, Animation

def gradient_descent_fd(obj, alpha, iteraciones, h=1e-5, seed=None, x_inicial=None):
    """
    Gradiente descendente donde el gradiente se aproxima numéricamente
    mediante diferencias finitas centrales (no se deriva la función analíticamente).

    Parameters
    ----------
    obj : Objective_function
        Instancia con obj.d, obj.lb, obj.ub y obj.function(x) definidos.
    alpha : float
        Tasa de aprendizaje.
    iteraciones : int
        Número de pasos de descenso.
    h : float, opcional
        Tamaño del paso para la diferencia finita central.
    seed : int, opcional
        Semilla para el punto inicial aleatorio.
    x_inicial : np.array, opcional
        Si se da, se usa como punto de partida en vez de uno aleatorio.

    Returns
    -------
    best_x : np.array de forma (d,)
    best_f : float
    lotes : list de (X_lote, ig_local)
        Un lote por iteración (X_lote de forma (d, 1)), listo para Animation.
    """
    if seed is not None:
        np.random.seed(seed)


    if x_inicial is None:
        lb = np.array(obj.lb)
        ub = np.array(obj.ub)
        x = lb + np.random.rand(obj.d) * (ub - lb)
    else:
        x = np.array(x_inicial, dtype=float)

    lotes = []
    best_x = x.copy()
    best_f = obj.function(x)

    for it in range(iteraciones):
        grad = np.zeros(obj.d)
        for i in range(obj.d):
            x_mas = x.copy()
            x_menos = x.copy()
            x_mas[i] += h
            x_menos[i] -= h
            grad[i] = (obj.function(x_mas) - obj.function(x_menos)) / (2 * h)

        x = x - alpha * grad
        x = np.clip(x, obj.lb, obj.ub)

        f_x = obj.function(x)
        X_lote = x.reshape(obj.d, 1)   # una sola columna: un punto por iteración
        lotes.append((X_lote, 0))

        if f_x < best_f:
            best_f = f_x
            best_x = x.copy()

    return best_x, best_f, lotes


if __name__ == "__main__":
    from sphere import Sphere   

    obj = Sphere(dim=2)

    best_x, best_f, lotes = gradient_descent_fd(
        obj, alpha=0.1, iteraciones=100, h=1e-5, seed=42
    )

    print("Mejor x:", best_x)
    print("Mejor f(x):", best_f)
    print("Número de iteraciones (cuadros de animación):", len(lotes))

    anim = Animation(animation_type='3d', name="Gradient Descent (FD) - Sphere")
    X0, ig0 = lotes[0]
    anim.initialize_animation(obj, X0, ig0)
    for X_lote, ig in lotes[1:]:
        anim.update_animation(obj, X_lote, ig)

    plt.show()