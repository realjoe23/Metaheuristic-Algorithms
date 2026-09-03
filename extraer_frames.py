#%% Bibliotecas necesarias
import os
from PIL import Image

#%% Configuración
INPUT_DIR = "resultados"
OUTPUT_DIR = "resultados_png"
os.makedirs(OUTPUT_DIR, exist_ok=True)

algoritmos = ["exhaustive", "random", "gradient_descent", "hill_climbing"]
funciones = ["sphere", "rastringin", "ackley", "zakharov"]

#%% Función para extraer un frame de un GIF y guardarlo como PNG
def extraer_frame(ruta_gif, ruta_png, indice_frame=None):
    """
    Extrae un frame de un GIF animado y lo guarda como PNG.

    Parameters
    ----------
    ruta_gif : str
        Ruta al archivo .gif de entrada.
    ruta_png : str
        Ruta donde se guardará el .png de salida.
    indice_frame : int, opcional
        Índice del frame a extraer. Si es None, se usa el frame de en medio
        (buena elección por defecto: muestra "camino recorrido" sin ser
        el arranque ni el final).
    """
    gif = Image.open(ruta_gif)

    n_frames = gif.n_frames
    if indice_frame is None:
        indice_frame = n_frames // 2
    else:
        indice_frame = min(indice_frame, n_frames - 1)  # evitar índice fuera de rango

    gif.seek(indice_frame)
    gif.convert("RGB").save(ruta_png)


#%% Extraer las 16 combinaciones
for alg in algoritmos:
    for func in funciones:
        nombre_base = f"{alg}_{func}"
        ruta_gif = os.path.join(INPUT_DIR, f"{nombre_base}.gif")
        ruta_png = os.path.join(OUTPUT_DIR, f"{nombre_base}.png")

        if not os.path.exists(ruta_gif):
            print(f"AVISO: no se encontró {ruta_gif}, se omite.")
            continue

        extraer_frame(ruta_gif, ruta_png)
        print(f"Extraído: {ruta_png}")

print("\nListo. Todos los PNG están en la carpeta:", OUTPUT_DIR)