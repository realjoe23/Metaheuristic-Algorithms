#%% Bibliotecas necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

#%% Carga de datos
datos = pd.read_csv("Ajuste_de_curvas.csv", index_col=0)
X = datos["x"].values
Y = datos["y"].values
NUM_DATOS = len(X)

#%% Modelo a ajustar
# Al graficar los datos se observa una tendencia lineal creciente con dos
# oscilaciones superpuestas de distinta frecuencia y amplitud. Se propone
# entonces un modelo de tendencia lineal más dos componentes sinusoidales:
#
#   y_hat(x) = a + b*x + A1*sin(w1*x + phi1) + A2*sin(w2*x + phi2)
#
# El cromosoma es el vector de 8 parámetros reales theta = [a, b, A1, w1,
# phi1, A2, w2, phi2] que el algoritmo genético debe encontrar.
NUM_PARAMS = 8
NOMBRES_PARAMS = ["a", "b", "A1", "w1", "phi1", "A2", "w2", "phi2"]

# Límites de búsqueda para cada parámetro, elegidos a partir de la escala
# de los datos (x en [0,100], y en aproximadamente [70, 1670]):
#   a, b       -> ordenada al origen y pendiente de la tendencia lineal
#   A1, A2     -> amplitudes, acotadas por el rango de variación de y
#   w1, w2     -> frecuencias angulares (un periodo mínimo razonable de ~3)
#   phi1, phi2 -> fases, con sentido físico solo en [-pi, pi]
LB = np.array([-500, -20,   0,   0, -np.pi,   0,   0, -np.pi])
UB = np.array([ 500,  20, 500,   2,  np.pi, 500,   2,  np.pi])

def modelo(x, theta):
    a, b, A1, w1, phi1, A2, w2, phi2 = theta
    return a + b * x + A1 * np.sin(w1 * x + phi1) + A2 * np.sin(w2 * x + phi2)

#%% Función objetivo
def error_cuadratico_medio(theta):
    """MSE entre el modelo y los datos reales. Esto es lo que el AG minimiza."""
    prediccion = modelo(X, theta)
    return np.mean((Y - prediccion) ** 2)

#%% Parámetros del algoritmo genético
POP_SIZE = 200
GENERATIONS = 1500
TOURNAMENT_PERCENT = 0.04
TOURNAMENT_SIZE = max(2, int(POP_SIZE * TOURNAMENT_PERCENT))
PROB_MUTACION_GEN = 0.3          # probabilidad de mutar cada gen de un hijo
SIGMA_INICIAL = 0.2 * (UB - LB)  # desviación estándar inicial de la mutación, por gen
DECAIMIENTO_SIGMA = 0.995        # la mutación se vuelve más fina con las generaciones
ELITISM = True
SEED = 42

if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)

#%% Funciones de ayuda del algoritmo genético de codificación real
def crear_cromosoma():
    """Un cromosoma es un vector de 8 números reales dentro de sus límites."""
    return LB + np.random.rand(NUM_PARAMS) * (UB - LB)

def crear_poblacion():
    return [crear_cromosoma() for _ in range(POP_SIZE)]

def torneo(poblacion, costos):
    """Selección por torneo: gana el cromosoma de menor error."""
    participantes_idx = random.sample(range(POP_SIZE), TOURNAMENT_SIZE)
    costos_participantes = [costos[i] for i in participantes_idx]
    return poblacion[participantes_idx[int(np.argmin(costos_participantes))]]

#%% Operadores genéticos para codificación real
def cruza_aritmetica(padre1, padre2):
    """
    Cruzamiento aritmético: cada gen del hijo es una combinación lineal
    (ponderada por un alpha aleatorio por gen) de los genes de los padres.
    A diferencia del cruzamiento de un punto usado en representaciones
    binarias u ordinales, este operador es el adecuado para cromosomas de
    números reales, ya que genera descendientes dentro del segmento que une
    a ambos padres en el espacio de parámetros.
    """
    alpha = np.random.rand(NUM_PARAMS)
    hijo1 = alpha * padre1 + (1 - alpha) * padre2
    hijo2 = alpha * padre2 + (1 - alpha) * padre1
    return hijo1, hijo2

def mutacion_gaussiana(cromosoma, sigma):
    """
    Mutación gaussiana: a cada gen, con probabilidad PROB_MUTACION_GEN, se le
    suma ruido N(0, sigma_i). sigma va disminuyendo con las generaciones
    (ver DECAIMIENTO_SIGMA), de modo que el algoritmo explora ampliamente al
    inicio y refina la solución hacia el final, de forma análoga a un
    enfriamiento en recocido simulado.
    """
    cromosoma = cromosoma.copy()
    for i in range(NUM_PARAMS):
        if random.random() < PROB_MUTACION_GEN:
            cromosoma[i] += np.random.normal(0, sigma[i])
    return np.clip(cromosoma, LB, UB)

#%% Algoritmo genético
def algoritmo_genetico():
    """
    Regresa:
        mejor_theta_global : mejor vector de parámetros encontrado
        mejor_mse_global    : su error cuadrático medio
        hist_mejor          : mejor MSE de cada generación
        hist_promedio       : MSE promedio de cada generación
    """
    poblacion = crear_poblacion()
    sigma = SIGMA_INICIAL.copy()

    hist_mejor = []
    hist_promedio = []
    mejor_theta_global = None
    mejor_mse_global = np.inf

    for g in range(GENERATIONS):
        # 1. Evaluar toda la población
        costos = [error_cuadratico_medio(c) for c in poblacion]

        # 2. Mejor de la generación
        mejor_idx = int(np.argmin(costos))
        mejor_cromosoma = poblacion[mejor_idx].copy()
        mejor_costo = costos[mejor_idx]

        hist_mejor.append(mejor_costo)
        hist_promedio.append(float(np.mean(costos)))

        # 3. Actualizar el mejor global
        if mejor_costo < mejor_mse_global:
            mejor_mse_global = mejor_costo
            mejor_theta_global = mejor_cromosoma.copy()

        # 4. Generar hijos: selección por torneo + cruza aritmética + mutación
        hijos = []
        while len(hijos) < POP_SIZE:
            padre1 = torneo(poblacion, costos)
            padre2 = torneo(poblacion, costos)

            hijo1, hijo2 = cruza_aritmetica(padre1, padre2)
            hijo1 = mutacion_gaussiana(hijo1, sigma)
            hijo2 = mutacion_gaussiana(hijo2, sigma)

            hijos.append(hijo1)
            if len(hijos) < POP_SIZE:
                hijos.append(hijo2)

        # 5. Elitismo: el mejor de la generación sobrevive sin modificarse
        if ELITISM:
            hijos[0] = mejor_cromosoma.copy()

        poblacion = hijos
        sigma *= DECAIMIENTO_SIGMA   # enfriamiento: mutación cada vez más fina

        if g % 100 == 0:
            print(f"Generacion {g}: Mejor MSE = {mejor_costo:.4f}")

    return mejor_theta_global, mejor_mse_global, hist_mejor, hist_promedio

#%% Gráficas
def graficar_ajuste(theta, nombre_archivo):
    """Compara los datos originales contra la curva generada por el modelo ajustado."""
    x_fino = np.linspace(X.min(), X.max(), 1000)
    y_fino = modelo(x_fino, theta)

    plt.figure(figsize=(9, 5))
    plt.scatter(X, Y, s=10, color='tab:gray', label='Datos (Ajuste_de_curvas.csv)')
    plt.plot(x_fino, y_fino, color='tab:red', linewidth=2, label='Curva ajustada por el AG')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Ajuste de curvas con algoritmo genético')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(nombre_archivo, dpi=200, bbox_inches='tight')

def graficar_historial(hist_mejor, hist_promedio, nombre_archivo):
    plt.figure(figsize=(8, 5))
    plt.plot(hist_promedio, label='MSE promedio de la población', color='tab:orange', alpha=0.8)
    plt.plot(hist_mejor, label='Mejor MSE de la generación', color='tab:blue', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Generación')
    plt.ylabel('Error cuadrático medio (escala log)')
    plt.title('Historial de la función de costo')
    plt.legend()
    plt.grid(alpha=0.3, which='both')
    plt.savefig(nombre_archivo, dpi=200, bbox_inches='tight')

#%% Programa principal
if __name__ == "__main__":
    mejor_theta, mejor_mse, hist_mejor, hist_promedio = algoritmo_genetico()

    prediccion = modelo(X, mejor_theta)
    ss_res = np.sum((Y - prediccion) ** 2)
    ss_tot = np.sum((Y - Y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print("\nMejores parámetros encontrados:")
    for nombre, valor in zip(NOMBRES_PARAMS, mejor_theta):
        print(f"  {nombre} = {valor:.6f}")
    print(f"MSE final: {mejor_mse:.6f}")
    print(f"RMSE final: {np.sqrt(mejor_mse):.6f}")
    print(f"R^2: {r2:.8f}")
    print(f"MSE de la población inicial: {hist_mejor[0]:.4f}")

    graficar_ajuste(mejor_theta, "ajuste_curva.png")
    graficar_historial(hist_mejor, hist_promedio, "historial_costo_ajuste.png")
    plt.show()