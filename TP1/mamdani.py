import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


# variables universales
x_nota = np.arange(0, 105, 5)
y_concepto = np.arange(0, 10.5, 0.5)
a_final = np.arange(0, 105, 5)

#! paso 1: FUZZIFICACION
# FUNCIONES DE MEMBRESIA DIFUSAS

nota_low = fuzz.trimf(x_nota, [0, 0, 50])  # nota baja: [0,30]
nota_md = fuzz.trimf(x_nota, [30, 55, 80])  # nota media: [40,60]
nota_hi = fuzz.trimf(x_nota, [60, 100, 100])  # nota alta [70,100]

concepto_low = fuzz.gaussmf(y_concepto, 0, 1.5)
concepto_md = fuzz.gaussmf(y_concepto, 7, 1.8)
concepto_hi = fuzz.gaussmf(y_concepto, 10, 1.5)

final_low = fuzz.trimf(a_final, [0, 0, 40])
final_md = fuzz.trimf(a_final, [30, 50, 70])
final_hi = fuzz.trimf(a_final, [60, 100, 100])


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

value_nota = 70
value_concepto = 3
peso_nota = 8
peso_concepto = 2


print("nota examen:", value_nota)
print("nota concepto:", value_concepto)
print("")


# CALCULA EL GRADO DE MEMBRESIA DE UN VALOR DADO EN UNA FUNCION DE MEM DIFUSA
nota_level_lo = fuzz.interp_membership(x_nota, nota_low, value_nota)
nota_level_md = fuzz.interp_membership(x_nota, nota_md, value_nota)
nota_level_hi = fuzz.interp_membership(x_nota, nota_hi, value_nota)

concepto_level_lo = fuzz.interp_membership(y_concepto, concepto_low, value_concepto)
concepto_level_md = fuzz.interp_membership(y_concepto, concepto_md, value_concepto)
concepto_level_hi = fuzz.interp_membership(y_concepto, concepto_hi, value_concepto)


print("nota_level_lo", nota_level_lo, "concepto_level_lo", concepto_level_lo)
print("nota_level_md", nota_level_md, "concepto_level_lo", concepto_level_md)
print("nota_level_hi", nota_level_hi, "concepto_level_lo", concepto_level_hi)
print("")


# rule 1: nota baja OR concepto regular then recursa
active_rule1 = (peso_nota * nota_level_lo + peso_concepto * concepto_level_lo) / (
    peso_nota + peso_concepto
)
print("active_rule1", active_rule1)
final_activation_lo = np.fmin(active_rule1, final_low)

# rule 2: nota media then habilita #!eliminamos el peso del concepto
active_rule2 = nota_level_md
print("active_rule3", active_rule2)
final_activation_md = np.fmin(active_rule2, final_md)

# rule 3: nota alta OR concepto alto then promociona
active_rule3 = (peso_nota * nota_level_hi + peso_concepto * concepto_level_hi) / (
    peso_nota + peso_concepto
)
# subrule2 = (peso_nota * nota_level_hi + peso_concepto * concepto_level_md) / (
#   peso_nota + peso_concepto
# )
# active_rule3 = np.fmin(subrule1, subrule2)
print("active_rule3", active_rule3)
final_activation_hi = np.fmin(active_rule3, final_hi)


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


#! paso 4: DEFUZZIFICACION

final = fuzz.defuzz(a_final, aggregated, "centroid")
final_activation = fuzz.interp_membership(a_final, aggregated, final)
print(f"------------ NOTA FINAL:{final/10} --------------")


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
