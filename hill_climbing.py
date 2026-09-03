import numpy as np
import matplotlib.pyplot as plt
from basics import Objective_function, Animation

def hill_climbing(obj, iteraciones, step_size=0.1, seed=None, x_inicial=None):
    """
    Hill Climbing simple: en cada iteración genera UN vecino aleatorio
    cercano al punto actual, y se mueve ahí solo si mejora.

    Parameters
    ----------
    obj : Objective_function
        Instancia con obj.d, obj.lb, obj.ub y obj.function(x) definidos.
    iteraciones : int
        Número de iteraciones (intentos de vecino).
    step_size : float, opcional
        Desviación estándar del ruido gaussiano usado para generar el vecino.
    seed : int, opcional
        Semilla para reproducibilidad.
    x_inicial : np.array, opcional
        Si se da, se usa como punto de partida en vez de uno aleatorio.

    Returns
    -------
    best_x : np.array de forma (d,)
    best_f : float
    lotes : list de (X_lote, ig_local)
        Un lote por iteración (X_lote de forma (d, 1)), listo para Animation.
        Nota: aquí X_lote siempre es el punto ACTUAL (se haya movido o no
        en esa iteración), para que la animación muestre también los
        intentos donde se quedó quieto.
    """
    if seed is not None:
        np.random.seed(seed)

    lb = np.array(obj.lb)
    ub = np.array(obj.ub)


    if x_inicial is None:
        x = lb + np.random.rand(obj.d) * (ub - lb)
    else:
        x = np.array(x_inicial, dtype=float)

    f_x = obj.function(x)

    lotes = []
    best_x = x.copy()
    best_f = f_x

    for it in range(iteraciones):
        vecino = x + np.random.normal(0, step_size, obj.d)
        vecino = np.clip(vecino, lb, ub)  # no salirse del dominio

        f_vecino = obj.function(vecino)

        if f_vecino < f_x:
            x = vecino
            f_x = f_vecino

        X_lote = x.reshape(obj.d, 1)
        lotes.append((X_lote, 0))

        if f_x < best_f:
            best_f = f_x
            best_x = x.copy()

    return best_x, best_f, lotes


if __name__ == "__main__":
    from sphere import Sphere   
    obj = Sphere(dim=2)

    best_x, best_f, lotes = hill_climbing(
        obj, iteraciones=200, step_size=0.2, seed=42
    )

    print("Mejor x:", best_x)
    print("Mejor f(x):", best_f)
    print("Número de iteraciones (cuadros de animación):", len(lotes))

    anim = Animation(animation_type='3d', name="Hill Climbing - Sphere")
    X0, ig0 = lotes[0]
    anim.initialize_animation(obj, X0, ig0)
    for X_lote, ig in lotes[1:]:
        anim.update_animation(obj, X_lote, ig)

    plt.show()