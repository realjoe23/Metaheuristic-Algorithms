import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

from sphere import Sphere 
from rastringin import Rastringin
from ackley import Ackley
from zakharov import Zakharov

from exhaustive_search import exhaustive_search
from random_search import random_search
from gradient_descent_fd import gradient_descent_fd
from hill_climbing import hill_climbing

OUTPUT_DIR = "resultados"
os.makedirs(OUTPUT_DIR, exist_ok=True)

funciones = {
    "sphere": Sphere,
    "rastringin": Rastringin,
    "ackley": Ackley,
    "zakharov": Zakharov,
}

alphas_gd = {
    "sphere": 0.1,
    "rastringin": 0.01,
    "ackley": 0.01,
    "zakharov": 0.001,
}

def crear_animacion_gif(obj, lotes, ruta_archivo, titulo):
    fig = plt.figure(figsize=(8, 8))
    obj.draw3d(resolution=100, name=titulo)
    ax = fig.gca()

    X0, ig0 = lotes[0]
    puntos, = ax.plot(X0[0, :], X0[1, :], obj.eval(X0), '.b')
    mejor, = ax.plot([X0[0, ig0]], [X0[1, ig0]], [obj.function(X0[:, ig0])],
                      '.r', markersize=12)

    def actualizar(frame):
        X_lote, ig = lotes[frame]
        puntos.set_data(X_lote[0, :], X_lote[1, :])
        puntos.set_3d_properties(obj.eval(X_lote))
        mejor.set_data([X_lote[0, ig]], [X_lote[1, ig]])
        mejor.set_3d_properties([obj.function(X_lote[:, ig])])
        return puntos, mejor

    anim = FuncAnimation(fig, actualizar, frames=len(lotes), interval=150)
    anim.save(ruta_archivo, writer=PillowWriter(fps=6))
    plt.close(fig)

def correr_algoritmo(nombre_alg, obj, nombre_func):
    if nombre_alg == "exhaustive":
        return exhaustive_search(obj, n_points=40)
    elif nombre_alg == "random":
        return random_search(obj, n_samples=1600, seed=42)
    elif nombre_alg == "gradient_descent":
        alpha = alphas_gd[nombre_func]
        return gradient_descent_fd(obj, alpha=alpha, iteraciones=200, h=1e-5, seed=42)
    elif nombre_alg == "hill_climbing":
        return hill_climbing(obj, iteraciones=200, step_size=0.2, seed=42)

algoritmos = ["exhaustive", "random", "gradient_descent", "hill_climbing"]
resumen = []

for nombre_func, ClaseFuncion in funciones.items():
    obj = ClaseFuncion(dim=2)

    for nombre_alg in algoritmos:
        print(f"Corriendo {nombre_alg} sobre {nombre_func}...")

        best_x, best_f, lotes = correr_algoritmo(nombre_alg, obj, nombre_func)

        titulo = f"{nombre_alg} - {nombre_func}"
        ruta = os.path.join(OUTPUT_DIR, f"{nombre_alg}_{nombre_func}.gif")
        crear_animacion_gif(obj, lotes, ruta, titulo)

        resumen.append({
            "algoritmo": nombre_alg,
            "funcion": nombre_func,
            "best_x": best_x,
            "best_f": best_f,
        })

print("\n=== RESUMEN ===")
for r in resumen:
    print(f"{r['algoritmo']:18s} | {r['funcion']:12s} | best_f = {r['best_f']:.6e}")