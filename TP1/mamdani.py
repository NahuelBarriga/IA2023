import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


"""
x1 nota baja    y1 concepto regular     a1 recursa
x2 media        y2 bueno                a2 habilita
x3 alta         y3 excelente            a3 promociona

regla 1:
- x1 and y1 then a1

regla 2:
- x2 and y1 then a2
- x3 and y2 then a2

regla 3:
- x3 and y3 then a2
- x3 and y2 then a2
- x2 and y3 then a2 

"""


# [ x and  A --> B  ] = [Umin(x,A) --> B ]

# variables universales
x_nota = np.arange(0, 100, 5)
y_concepto = np.arange(0, 10.5, 0.5)
a_final = np.arange(0, 10.5, 0.5)

"""
regular = np.arange(0,3,1)
bueno = np.arange(4,6,1)
excelente = np.arange(7,10,1)
"""


#! paso 1: FUZZIFICACION
# FUNCIONES DE MEMBRESIA DIFUSAS

# ?chequear estos parametros

nota_low = fuzz.trapmf(x_nota, [0, 0, 20, 40])  # nota baja: [0,30]
nota_md = fuzz.trapmf(x_nota, [20, 40, 60, 80])  # nota media: [40,60]
nota_hi = fuzz.trapmf(x_nota, [60, 80, 100, 100])  # nota alta [70,100]

concepto_low = fuzz.trapmf(y_concepto, [0, 0, 3, 3])
concepto_md = fuzz.trapmf(y_concepto, [4, 4, 6, 6])
concepto_hi = fuzz.trapmf(y_concepto, [7, 7, 10, 10])

final_low = fuzz.trapmf(a_final, [0, 0, 2, 4])
final_md = fuzz.trapmf(a_final, [2, 4, 6, 8])
final_hi = fuzz.trapmf(a_final, [6, 8, 10, 10])

# grafico de estas funciones
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(x_nota, nota_low, "b", linewidth=1.5, label="baja")
ax0.plot(x_nota, nota_md, "g", linewidth=1.5, label="media")
ax0.plot(x_nota, nota_hi, "r", linewidth=1.5, label="alta")
ax0.set_title("nota examen")
ax0.legend()

ax1.plot(y_concepto, concepto_low, "b", linewidth=1.5, label="regular")
ax1.plot(y_concepto, concepto_md, "g", linewidth=1.5, label="bueno")
ax1.plot(y_concepto, concepto_hi, "r", linewidth=1.5, label="excelente")
ax1.set_title("nota concepto")
ax1.legend()

ax2.plot(a_final, final_low, "b", linewidth=1.5, label="recursa")
ax2.plot(a_final, final_md, "g", linewidth=1.5, label="habilita")
ax2.plot(a_final, final_hi, "r", linewidth=1.5, label="promociona")
ax2.set_title("nota final")
ax2.legend()

# para mejorar la apariencia de los graficos
for ax in (ax0, ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

#! paso 2: INFERENCIA

value_nota = 90
value_concepto = "Regular" # "Regular" "Buena" "Excelente"
peso_nota = 5
peso_concepto = 1

metodo_defuzz = "bisector"

print("nota examen:", value_nota)
print("nota concepto:", value_concepto)
print("")

if (value_concepto == "Regular"):
    value_concepto = 2
elif (value_concepto == "Buena"):
    value_concepto = 4
else: 
    value_concepto = 10

prom_ponderado = (peso_nota * value_nota + peso_concepto * value_concepto) / (
    peso_nota + peso_concepto
)

# CALCULA EL GRADO DE MEMBRESIA DE UN VALOR DADO EN UNA FUNCION DE MEM DIFUSA
nota_level_lo = fuzz.interp_membership(x_nota, nota_low, value_nota)
nota_level_md = fuzz.interp_membership(x_nota, nota_md, value_nota)
nota_level_hi = fuzz.interp_membership(x_nota, nota_hi, value_nota)

concepto_level_lo = fuzz.interp_membership(y_concepto, concepto_low, value_concepto)
concepto_level_md = fuzz.interp_membership(y_concepto, concepto_md, value_concepto)
concepto_level_hi = fuzz.interp_membership(y_concepto, concepto_hi, value_concepto)

# rule evaluation: [ x and  y --> A  ] = [Umin(x,y) --> A ]

"""
The second step is to take the fuzzified inputs, and apply them to the antecedents
of the fuzzy rules. If a given fuzzy rule has multiple antecedents, the
fuzzy operator (AND or OR) is used to obtain a single number that
represents the result of the antecedent evaluation
"""

# final_activacion_ii: valores de activacion de diferentes reglas
# ? (fuerza de disparo??)

print("nota_level_lo", nota_level_lo, "concepto_level_lo", concepto_level_lo)
print("nota_level_lo", nota_level_md, "concepto_level_lo", concepto_level_md)
print("nota_level_lo", nota_level_hi, "concepto_level_lo", concepto_level_hi)
print("")


# rule 1: nota baja OR concepto regular then recursa
active_rule1 = (peso_nota * nota_level_lo + peso_concepto * concepto_level_lo) / (
    peso_nota + peso_concepto
)
print("active_rule1", active_rule1)
final_activation_lo = np.fmin(active_rule1, final_low)

# rule 2: nota media OR concepto bueno then habilita #!eliminamos el peso del concepto
"""
active_rule2 = (peso_nota * nota_level_md + peso_concepto * concepto_level_md) / (
    peso_nota + peso_concepto
)
"""
active_rule2 = nota_level_md
print("active_rule3", active_rule2)

final_activation_md = np.fmin(active_rule2, final_md)

# rule 3: nota alta OR concepto alto then promociona
active_rule3 = (peso_nota * nota_level_hi + peso_concepto * concepto_level_hi) / (
    peso_nota + peso_concepto
)
print("active_rule3", active_rule3)
final_activation_hi = np.fmin(active_rule3, final_hi)


print("")
print("final: desaprobar", final_activation_lo)
print("final: habilitar", final_activation_md)
print("finial: promocionar", final_activation_hi)
print("")

final0 = np.zeros_like(a_final)


# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(a_final, final0, final_activation_lo, facecolor="b", alpha=0.7)
ax0.plot(
    a_final,
    final_low,
    "b",
    linewidth=0.5,
    linestyle="--",
)
ax0.fill_between(a_final, final0, final_activation_md, facecolor="g", alpha=0.7)
ax0.plot(a_final, final_md, "g", linewidth=0.5, linestyle="--")
ax0.fill_between(a_final, final0, final_activation_hi, facecolor="r", alpha=0.7)
ax0.plot(a_final, final_hi, "r", linewidth=0.5, linestyle="--")
ax0.set_title("INFERENCIA")

# Turn off top/right axes
for ax in (ax0,):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

#! paso 3: AGREGACION

aggregated = np.fmax(
    final_activation_lo, np.fmax(final_activation_md, final_activation_hi)
)
print("agregated", aggregated)

#! paso 4: DEFUZZIFICACION

final = fuzz.defuzz(a_final, aggregated, metodo_defuzz)
final_activation = fuzz.interp_membership(a_final, aggregated, final)
print(f"------------ NOTA FINAL:{final} --------------")


fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(
    a_final,
    final_low,
    "b",
    linewidth=0.5,
    linestyle="--",
)
ax0.plot(a_final, final_md, "g", linewidth=0.5, linestyle="--")
ax0.plot(a_final, final_hi, "r", linewidth=0.5, linestyle="--")
ax0.fill_between(a_final, final0, aggregated, facecolor="Orange", alpha=0.7)
ax0.plot([final, final], [0, final_activation], "k", linewidth=1.5, alpha=0.9)
ax0.set_title("Aggregated membership and result (line)")

# Turn off top/right axes
for ax in (ax0,):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

plt.show()