import random
from collections import namedtuple


def funcion_target(x):
    return 300 - (x - 15) ** 2


def decode_binary(binary):
    return int(binary, 2)


cant_individuos = 6
cant_bits = 5
prob_mutacion = 0.1  # ? en base a qué?
num_generaciones = 100
elite_size = 2
cant_hijos = 4
crossover_points = (2, 3)
bit_to_mutate = 2

Poblacion = namedtuple("Poblacion", ["binary", "fitness", "ratio"])


#!Paso 1:
# Crear aleatoriamente los individuos que compondrán la población inicial
# ? estos individuos pueden ser repetidos??

poblacion_inicial = [
    Poblacion(binary=format(random.randint(0, 31), "05b"), fitness=0, ratio=0)
    for _ in range(cant_individuos)
]
print("Poblacion inicial")
print(poblacion_inicial)
print("")
totTarget = 0

for generacion in range(num_generaciones):
    #! Paso 2:
    # Calcular para cada uno el valor de la función de evaluación 𝑓𝑖(𝑥) y
    # la probabilidad de selección 𝑓𝑖(𝑥)/ ∑𝑘 𝑓𝑘(𝑥). Reconocer el mejor individuo de
    # esta población inicial
    i = 0
    for i, individuo in enumerate(poblacion_inicial):
        valor_int = decode_binary(
            individuo.binary
        )  #!EL ERROR ESTA ACA, NO ENTIENDO POR QUEEEEEEEE :( :(
        # print("valor int", valor_int, "valor i", i)
        fitness = funcion_target(valor_int)
        poblacion_inicial[i] = Poblacion(
            binary=individuo.binary, fitness=fitness, ratio=0
        )
        totTarget += fitness

    total_fitness = sum(individuo.fitness for individuo in poblacion_inicial)
    for i, individuo in enumerate(poblacion_inicial):
        ratio = individuo.fitness / total_fitness
        poblacion_inicial[i] = Poblacion(
            binary=individuo.binary, fitness=individuo.fitness, ratio=ratio
        )

    elite = sorted(
        poblacion_inicial, key=lambda individuo: individuo.fitness, reverse=True
    )[:elite_size]

    #!Paso 3:
    # Se configura que 2 individuos pasarán por elite a la generación siguiente. Además, los dos mejores individuos generarán 4
    # hijos por cruzamiento, con puntos de cruce en los bits
    #  2 y 3. Generar los hijos.

    hijos = []
    for generacion in range(cant_hijos):
        padre1 = random.choice(elite)
        padre2 = random.choice(elite)
        pto_cruce = random.randint(*crossover_points)
        hijo = padre1.binary[:pto_cruce] + padre2.binary[:pto_cruce]

        #!Paso 4:
        # Suponer que según la probabilidad de mutación dada se cambia el bit 2 del tercer hijo. Implementarla.
        if (generacion == 3) and (random.random() < prob_mutacion):
            if hijo[bit_to_mutate] == "0":
                hijo_lista = list(hijo)
                hijo_lista[bit_to_mutate] = "1"  # Cambia el bit
                hijo = "".join(hijo_lista)  # Convierte de nuevo a cadena binaria
            else:
                hijo_lista = list(hijo)
                hijo_lista[bit_to_mutate] = "0"  # Cambia el bit
                hijo = "".join(hijo_lista)  # Convierte de nuevo a cadena binaria

        hijos.append(hijo)
        poblacion_inicial = elite + hijos


max_fitness_elite = max(elite, key=lambda individuo: individuo.fitness)
print("Máximo de la función target en la élite:", max_fitness_elite.fitness)


x = totTarget / (num_generaciones * cant_hijos)
print("Promedio funcion target:", x)
