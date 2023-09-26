import random
from collections import namedtuple


def funcion_target(x):
    return 300 - (x - 15) ** 2


cant_individuos = 6
cant_bits = 5
prob_mutacion = 0.1  # ? en base a qué?
num_generaciones = 100

Poblacion = namedtuple("Poblacion", ["integer", "fitness", "ratio"])


#!Paso 1:
# Crear aleatoriamente los individuos que compondrán la población inicial
# ? estos individuos pueden ser repetidos??

poblacion_inicial = [
    Poblacion(integer=random.randint(0, 31), fitness=0, ratio=0)
    for _ in range(cant_individuos)
]


#! Paso 2:
# Calcular para cada uno el valor de la función de evaluación 𝑓𝑖(𝑥) y
# la probabilidad de selección 𝑓𝑖(𝑥)/ ∑𝑘 𝑓𝑘(𝑥). Reconocer el mejor individuo de
# esta población inicial

for generacion in range(num_generaciones):
    for i, individuo in enumerate(poblacion_inicial):
        fitness = funcion_target(individuo.integer)
        poblacion_inicial[i] = Poblacion(
            integer=individuo.integer, fitness=fitness, ratio=0
        )

    total_fitness = sum(individuo.fitness for individuo in poblacion_inicial)

    for i, individuo in enumerate(poblacion_inicial):
        ratio = individuo.fitness / total_fitness
        poblacion_inicial[i] = Poblacion(
            integer=individuo.integer, fitness=individuo.fitness, ratio=ratio
        )

# Imprimir información de la población
for individuo in poblacion_inicial:
    print("Valor:", individuo.integer)
    print("Fitness:", individuo.fitness)
    print("Ratio:", individuo.ratio)
    print()

best_individuo = max(poblacion_inicial, key=lambda individuo: individuo.fitness)
# Imprime la información del mejor individuo
print("Mejor Individuo:")
print("Valor:", best_individuo.integer)
print("Fitness:", best_individuo.fitness)
print("Ratio:", best_individuo.ratio)


#!Paso 3:
# Se configura que 2 individuos pasarán por elite a la generación siguiente. Además, los dos mejores individuos generarán 4
# hijos por cruzamiento, con puntos de cruce en los bits 2 y 3. Generar los hijos.

nueva_poblacion = []
