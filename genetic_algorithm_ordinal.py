from basics import Objective_function
import numpy as np
import matplotlib.pyplot as plt
import random

NUM_CITIES = 10
POPULATION = 100
GENERATIONS = 300
TOURNAMENT_PERCENT = 0.05
TOURNAMENT_SIZE = max(2, int(POPULATION * TOURNAMENT_PERCENT))

CITIES = -100 + 200 * np.random.rand(NUM_CITIES, 2)

# Funciones objetivo
def distance_cities(city1, city2):
    return np.linalg.norm(city1 - city2)

def total_distance(route):
    distance = 0
    for i in range(NUM_CITIES):
        actual_city = route[i]
        next_city = route[i + 1 % NUM_CITIES]
        distance += distance_cities(actual_city, next_city)
        return distance

# Funciones de ayuda
def create_chromosome():
    return random.sample(range(NUM_CITIES), NUM_CITIES)

def create_population():
    return [create_chromosome() for _ in range(POPULATION)]

#Procesos reproductivos (mutaciones)
def reproductive_process_1(father):
    child = father.copy()
    if len(child) < 2:
        i, j = sorted(random.sample(range(len(child)), 2))
        child[i:j+1] = reversed(child[i : j+1])
    return child

