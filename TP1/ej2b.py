import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x_humedad = np.arange(0.5, 4.1, 0.1)
y_malezas = np.arange(0, 1.1, 0.1)
z_horas = np.arange(0.5, 4.1, 0.1)

humedad_mf_baja = fuzz.trapmf(x_humedad, [0.5, 0.5, 1.5, 2.5])
humedad_mf_alta = fuzz.trapmf(x_humedad, [1.5, 3, 4, 4])

maleza_mf_esc = fuzz.trapmf(y_malezas, [0, 0, 0.3, 0.6])
maleza_mf_ab = fuzz.trapmf(y_malezas, [0.4, 0.5, 1, 1])

horas_mf_pocas = fuzz.trapmf(z_horas, [0.5, 0.5, 1.8, 3])
horas_mf_muchas = fuzz.trapmf(z_horas, [1.5, 3, 4, 4])

# Visualize these universes and membership functions
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

# value_x = 8
# value_x = -8
# value_x = 5
value_humedad = 1.87
value_malotes = 0.434


ax0.plot(x_humedad, humedad_mf_baja, "b", linewidth=1.5, label="Baja")
ax0.plot(x_humedad, humedad_mf_alta, "g", linewidth=1.5, label="Alta")

ax0.set_title("Humedad")
ax0.legend()

ax1.plot(y_malezas, maleza_mf_esc, "b", linewidth=1.5, label="Escasa")
ax1.plot(y_malezas, maleza_mf_ab, "g", linewidth=1.5, label="Abundante")
ax1.set_title("Malezas")
ax1.legend()

ax2.plot(z_horas, horas_mf_pocas, "g", linewidth=1.5, label="Pocas")
ax2.plot(z_horas, horas_mf_muchas, "b", linewidth=1.5, label="Muchas")
ax2.set_title("Horas")
ax2.legend()

# INFERENCE

# We need the activation of our fuzzy membership functions at these values.
# The exact values 6.5 and 9.8 do not exist on our universes...
# This is what fuzz.interp_membership exists for!

humedad_baja = fuzz.interp_membership(x_humedad, humedad_mf_baja, value_humedad)
humedad_alta = fuzz.interp_membership(x_humedad, humedad_mf_alta, value_humedad)

malezas_esc = fuzz.interp_membership(y_malezas, maleza_mf_esc, value_malotes)
malezas_ab = fuzz.interp_membership(y_malezas, maleza_mf_ab, value_malotes)

# Rule 1
horas_muchas = np.fmin(np.fmax(humedad_baja, malezas_ab), horas_mf_muchas)

# Rule 2
horas_pocas = np.fmin(np.fmin(humedad_alta, malezas_esc), horas_mf_pocas)


# pone un piso para rellenar la funcion truncada
salida = np.zeros_like(z_horas)

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(z_horas, salida, horas_muchas, facecolor="b", alpha=0.7)
ax0.plot(
    z_horas,
    horas_muchas,
    "b",
    linewidth=0.5,
    linestyle="--",
)
ax0.fill_between(z_horas, salida, horas_pocas, facecolor="g", alpha=0.7)
ax0.plot(z_horas, horas_pocas, "g", linewidth=0.5, linestyle="--")
ax0.set_title("INFERENCIA")
plt.tight_layout()

# AGREGATION
aggregated = np.fmax(horas_muchas, horas_pocas)

# DEFUZZIFICATION
horas_final = fuzz.defuzz(z_horas, aggregated, "centroid")
print(f"VALOR DE Y = {horas_final}")
y_activation = fuzz.interp_membership(z_horas, aggregated, horas_final)

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(
    z_horas,
    horas_pocas,
    "b",
    linewidth=0.5,
    linestyle="--",
)
ax0.plot(z_horas, horas_muchas, "g", linewidth=0.5, linestyle="--")
ax0.fill_between(z_horas, salida, aggregated, facecolor="Orange", alpha=0.7)
ax0.plot([horas_final, horas_final], [0, y_activation], "k", linewidth=1.5, alpha=0.9)
ax0.set_title("AGREGACION")

plt.show()
