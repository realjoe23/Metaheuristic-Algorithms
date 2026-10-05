#%% Bibliotecas necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

#%% Carga de datos
datos = pd.read_csv("mochila.csv")
PESOS = datos["peso"].values
PRECIOS = datos["precio"].values
NUM_ITEMS = len(PESOS)

# Capacidad de la mochila: no fue especificada en la tarea, así que se fija
# como el 50% del peso total de todos los objetos. Esto obliga a descartar
# aproximadamente la mitad del valor disponible, lo cual hace el problema
# no trivial (si la capacidad fuera mayor a la suma de todos los pesos,
# la solución óptima sería simplemente llevar todo).
CAPACIDAD = 0.5 * PESOS.sum()

#%% Parámetros del algoritmo genético
POP_SIZE = 150
GENERATIONS = 300
TOURNAMENT_PERCENT = 0.05
TOURNAMENT_SIZE = max(2, int(POP_SIZE * TOURNAMENT_PERCENT))
PROB_CRUZA = 0.9
PROB_MUTACION = 1.0 / NUM_ITEMS   # en promedio, un gen mutado por cromosoma
ELITISM = True
SEED = 42

if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)

#%% Función objetivo (con reparación de soluciones no válidas)
def reparar(cromosoma):
    """
    Si el peso total excede la capacidad, se van quitando objetos hasta que
    el cromosoma sea válido. Se eliminan primero los objetos con peor razón
    precio/peso (los "menos eficientes"), ya que son los que menos aportan
    por unidad de peso que ocupan. Esto evita descartar soluciones completas
    por una pequeña violación de la restricción (a diferencia de una
    penalización en la función de costo) y mantiene siempre cromosomas
    factibles en la población.
    """
    peso_total = np.dot(cromosoma, PESOS)
    if peso_total <= CAPACIDAD:
        return cromosoma

    cromosoma = cromosoma.copy()
    indices_incluidos = [i for i in range(NUM_ITEMS) if cromosoma[i] == 1]
    # Ordenar los objetos incluidos de peor a mejor razón precio/peso
    indices_incluidos.sort(key=lambda i: PRECIOS[i] / PESOS[i])

    for i in indices_incluidos:
        if peso_total <= CAPACIDAD:
            break
        cromosoma[i] = 0
        peso_total -= PESOS[i]

    return cromosoma

def evaluar(cromosoma):
    """Valor total de los objetos incluidos (el cromosoma ya debe ser válido)."""
    return float(np.dot(cromosoma, PRECIOS))

#%% Funciones de ayuda del algoritmo genético binario
def crear_cromosoma():
    """Cromosoma binario: 1 si el objeto se incluye, 0 si no."""
    cromosoma = np.random.randint(0, 2, NUM_ITEMS)
    return reparar(cromosoma)

def crear_poblacion():
    return [crear_cromosoma() for _ in range(POP_SIZE)]

def torneo(poblacion, valores):
    """Selección por torneo: gana el cromosoma de mayor valor."""
    participantes_idx = random.sample(range(POP_SIZE), TOURNAMENT_SIZE)
    valores_participantes = [valores[i] for i in participantes_idx]
    return poblacion[participantes_idx[int(np.argmax(valores_participantes))]]

#%% Operadores genéticos
def cruza_un_punto(padre1, padre2):
    """Cruzamiento de un punto: intercambia los genes a partir de un corte aleatorio."""
    if random.random() > PROB_CRUZA:
        return padre1.copy(), padre2.copy()
    punto = random.randint(1, NUM_ITEMS - 1)
    hijo1 = np.concatenate([padre1[:punto], padre2[punto:]])
    hijo2 = np.concatenate([padre2[:punto], padre1[punto:]])
    return hijo1, hijo2

def mutacion(cromosoma):
    """Mutación bit-flip: cada gen tiene una probabilidad PROB_MUTACION de invertirse."""
    cromosoma = cromosoma.copy()
    for i in range(NUM_ITEMS):
        if random.random() < PROB_MUTACION:
            cromosoma[i] = 1 - cromosoma[i]
    return cromosoma

#%% Algoritmo genético
def algoritmo_genetico():
    """
    Regresa:
        mejor_cromosoma_global : mejor solución encontrada en todas las generaciones
        mejor_valor_global     : su valor total
        hist_mejor             : mejor valor de cada generación
        hist_promedio          : valor promedio de cada generación
    """
    poblacion = crear_poblacion()

    hist_mejor = []
    hist_promedio = []
    mejor_cromosoma_global = None
    mejor_valor_global = -np.inf

    for g in range(GENERATIONS):
        # 1. Evaluar toda la población
        valores = [evaluar(c) for c in poblacion]

        # 2. Mejor de la generación
        mejor_idx = int(np.argmax(valores))
        mejor_cromosoma = poblacion[mejor_idx].copy()
        mejor_valor = valores[mejor_idx]

        hist_mejor.append(mejor_valor)
        hist_promedio.append(float(np.mean(valores)))

        # 3. Actualizar el mejor global
        if mejor_valor > mejor_valor_global:
            mejor_valor_global = mejor_valor
            mejor_cromosoma_global = mejor_cromosoma.copy()

        # 4. Generar hijos: selección por torneo + cruza + mutación
        hijos = []
        while len(hijos) < POP_SIZE:
            padre1 = torneo(poblacion, valores)
            padre2 = torneo(poblacion, valores)

            hijo1, hijo2 = cruza_un_punto(padre1, padre2)
            hijo1 = reparar(mutacion(hijo1))
            hijo2 = reparar(mutacion(hijo2))

            hijos.append(hijo1)
            if len(hijos) < POP_SIZE:
                hijos.append(hijo2)

        # 5. Elitismo: el mejor de la generación sobrevive sin modificarse
        if ELITISM:
            hijos[0] = mejor_cromosoma.copy()

        poblacion = hijos

        if g % 50 == 0:
            print(f"Generacion {g}: Mejor valor = {mejor_valor:.2f}  "
                  f"Peso usado = {np.dot(mejor_cromosoma, PESOS):.2f} / {CAPACIDAD:.2f}")

    return mejor_cromosoma_global, mejor_valor_global, hist_mejor, hist_promedio

#%% Gráficas
def graficar_historial(hist_mejor, hist_promedio, nombre_archivo):
    plt.figure(figsize=(8, 5))
    plt.plot(hist_promedio, label='Promedio de la población', color='tab:orange', alpha=0.8)
    plt.plot(hist_mejor, label='Mejor de la generación', color='tab:blue', linewidth=2)
    plt.xlabel('Generación')
    plt.ylabel('Valor total de la mochila')
    plt.title('Historial de la función de costo')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(nombre_archivo, dpi=200, bbox_inches='tight')

def graficar_solucion(cromosoma, nombre_archivo):
    """
    Dispersión peso vs. precio de los 100 objetos, distinguiendo los
    incluidos en la mochila de los descartados.
    """
    incluidos = cromosoma == 1
    plt.figure(figsize=(7, 6))
    plt.scatter(PESOS[~incluidos], PRECIOS[~incluidos], color='lightgray',
                label='Descartado', alpha=0.8)
    plt.scatter(PESOS[incluidos], PRECIOS[incluidos], color='tab:green',
                label='Incluido', alpha=0.9)
    plt.xlabel('Peso')
    plt.ylabel('Precio')
    plt.title('Objetos incluidos en la mejor solución encontrada')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(nombre_archivo, dpi=200, bbox_inches='tight')

#%% Programa principal
if __name__ == "__main__":
    mejor_cromosoma, mejor_valor, hist_mejor, hist_promedio = algoritmo_genetico()

    peso_usado = np.dot(mejor_cromosoma, PESOS)
    num_incluidos = int(mejor_cromosoma.sum())

    print("\nMejor valor encontrado:", mejor_valor)
    print(f"Peso utilizado: {peso_usado:.2f} / {CAPACIDAD:.2f}")
    print(f"Objetos incluidos: {num_incluidos} de {NUM_ITEMS}")
    print(f"Mejor valor de la población inicial: {hist_mejor[0]:.2f}")

    graficar_historial(hist_mejor, hist_promedio, "historial_costo_mochila.png")
    graficar_solucion(mejor_cromosoma, "solucion_mochila.png")
    plt.show()