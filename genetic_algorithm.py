#%% Bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import random

#%% Parámetros del problema y del algoritmo
NUM_CITIES = 20                 # número de ciudades del problema
POP_SIZE = 100                  # tamaño de la población
GENERATIONS = 500               # número de generaciones
TOURNAMENT_PERCENT = 0.05       # porcentaje de la población que compite en cada torneo
TOURNAMENT_SIZE = max(2, int(POP_SIZE * TOURNAMENT_PERCENT))
ELITISM = True                  # si es True, el mejor cromosoma pasa intacto a la siguiente generación
SEED = 42                       # semilla para poder reproducir los resultados (None = aleatorio)

if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)

# Posición aleatoria de las ciudades en el cuadro [-100, 100] x [-100, 100]
CITIES = -100 + 200 * np.random.rand(NUM_CITIES, 2)

#%% Función objetivo
def distancia_ciudades(ciudad1, ciudad2):
    """Distancia euclidiana entre dos ciudades."""
    return np.linalg.norm(ciudad1 - ciudad2)

def distancia_total(ruta):
    """
    Longitud total de una ruta (ciclo cerrado): suma las distancias entre
    ciudades consecutivas y regresa de la última a la primera.
    """
    distancia = 0
    for i in range(NUM_CITIES):
        ciudad_actual = CITIES[ruta[i]]
        ciudad_siguiente = CITIES[ruta[(i + 1) % NUM_CITIES]]
        distancia += distancia_ciudades(ciudad_actual, ciudad_siguiente)
    return distancia

#%% Funciones de ayuda del algoritmo genético ordinal
def crear_cromosoma():
    """Un cromosoma es una permutación de las ciudades (representación ordinal)."""
    return random.sample(range(NUM_CITIES), NUM_CITIES)

def crear_poblacion():
    """Población inicial de POP_SIZE rutas aleatorias."""
    return [crear_cromosoma() for _ in range(POP_SIZE)]

#%% Procesos reproductivos (mutaciones que siempre producen permutaciones válidas)
def proceso_reversa(padre):
    """Invierte el orden de los genes entre dos posiciones aleatorias i y j."""
    hijo = padre.copy()
    if len(hijo) > 2:
        i, j = sorted(random.sample(range(len(hijo)), 2))
        hijo[i:j+1] = reversed(hijo[i:j+1])
    return hijo

def proceso_enroque(padre):
    """Extrae el gen de la posición i y lo inserta en la posición j."""
    hijo = padre.copy()
    if len(hijo) > 2:
        i, j = sorted(random.sample(range(len(hijo)), 2))
        gen = hijo.pop(i)
        hijo.insert(j, gen)
    return hijo

def torneo(poblacion, distancias):
    """Selección por torneo: gana la ruta más corta de TOURNAMENT_SIZE elegidas al azar."""
    participantes_idx = random.sample(range(POP_SIZE), TOURNAMENT_SIZE)
    distancias_participantes = [distancias[i] for i in participantes_idx]
    return poblacion[participantes_idx[int(np.argmin(distancias_participantes))]]

#%% Algoritmo genético ordinal
def algoritmo_genetico():
    """
    Regresa:
        mejor_ruta_global : mejor cromosoma encontrado en TODAS las generaciones
        mejor_dist_global : su distancia total
        hist_mejor        : mejor distancia de cada generación
        hist_promedio     : distancia promedio de cada generación
        ruta_inicial      : mejor ruta de la población inicial (para comparar)
    """
    poblacion = crear_poblacion()

    hist_mejor = []
    hist_promedio = []
    mejor_ruta_global = None
    mejor_dist_global = np.inf
    ruta_inicial = None

    for g in range(GENERATIONS):
        # 1. Evaluar toda la población
        distancias = [distancia_total(cromosoma) for cromosoma in poblacion]

        # 2. Mejor de la generación
        mejor_idx = int(np.argmin(distancias))
        mejor_cromosoma = poblacion[mejor_idx].copy()
        mejor_distancia = distancias[mejor_idx]

        hist_mejor.append(mejor_distancia)
        hist_promedio.append(float(np.mean(distancias)))

        if g == 0:
            ruta_inicial = mejor_cromosoma.copy()

        # 3. Actualizar el mejor global (evita perder la mejor ruta si no hay elitismo)
        if mejor_distancia < mejor_dist_global:
            mejor_dist_global = mejor_distancia
            mejor_ruta_global = mejor_cromosoma.copy()

        # 4. Generar los hijos: selección por torneo + mutación
        hijos = []
        for _ in range(POP_SIZE):
            ganador = torneo(poblacion, distancias)

            # Se elige al azar el proceso reproductivo
            if random.random() < 0.5:
                hijo = proceso_reversa(ganador)
            else:
                hijo = proceso_enroque(ganador)
            hijos.append(hijo)

        # 5. Elitismo: el mejor de la generación sobrevive sin modificarse
        if ELITISM:
            hijos[0] = mejor_cromosoma.copy()

        poblacion = hijos

        if g % 50 == 0:
            print(f"Generacion {g}: Mejor distancia = {mejor_distancia:.2f}")

    return mejor_ruta_global, mejor_dist_global, hist_mejor, hist_promedio, ruta_inicial

#%% Gráficas
def graficar_ruta(ruta, titulo, nombre_archivo):
    """Dibuja las ciudades y la ruta como ciclo cerrado."""
    coords = CITIES[ruta + [ruta[0]]]      # se agrega la primera ciudad al final para cerrar el ciclo
    plt.figure(figsize=(7, 7))
    plt.plot(coords[:, 0], coords[:, 1], 'o-', color='tab:blue')
    plt.plot(CITIES[ruta[0], 0], CITIES[ruta[0], 1], 's', color='tab:red',
             markersize=10, label='Ciudad de inicio')
    for k in range(NUM_CITIES):
        plt.annotate(str(k), (CITIES[k, 0], CITIES[k, 1]),
                     textcoords="offset points", xytext=(5, 5), fontsize=9)
    plt.title(titulo)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(nombre_archivo, dpi=200, bbox_inches='tight')

def graficar_historial(hist_mejor, hist_promedio, nombre_archivo):
    """Dibuja el historial de la función de costo (distancia total) por generación."""
    plt.figure(figsize=(8, 5))
    plt.plot(hist_promedio, label='Promedio de la población', color='tab:orange', alpha=0.8)
    plt.plot(hist_mejor, label='Mejor de la generación', color='tab:blue', linewidth=2)
    plt.xlabel('Generación')
    plt.ylabel('Distancia total')
    plt.title('Historial de la función de costo')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(nombre_archivo, dpi=200, bbox_inches='tight')

#%% Programa principal
if __name__ == "__main__":
    mejor_ruta, mejor_distancia, hist_mejor, hist_promedio, ruta_inicial = algoritmo_genetico()

    print("\nMejor ruta:", mejor_ruta)
    print(f"Mejor distancia: {mejor_distancia:.2f}")
    print(f"Mejor distancia de la población inicial: {hist_mejor[0]:.2f}")

    graficar_ruta(ruta_inicial, f"Mejor ruta de la población inicial (distancia = {hist_mejor[0]:.2f})",
                  "ruta_inicial.png")
    graficar_ruta(mejor_ruta, f"Mejor ruta encontrada (distancia = {mejor_distancia:.2f})",
                  "ruta_optima.png")
    graficar_historial(hist_mejor, hist_promedio, "historial_costo.png")
    plt.show()