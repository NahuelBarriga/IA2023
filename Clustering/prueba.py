import numpy as np
import os

data= []
if os.path.exists("Clustering\datos\datos.txt"): 
    print("hola") 
else:
    f=os.path.dirname("prueba.py")
    print("noh ola")



with open("clustering\datos.txt") as file:
     for line in file:
          values = line.split()
          row = [float(value) for value in values]
          data.append(row)
file.close()
print(data)
