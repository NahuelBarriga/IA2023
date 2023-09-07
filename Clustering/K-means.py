from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

datos= []
cantClusters=3

#* Pasa los datos del .txt a un array
with open("Clustering/datos/datos.txt") as file:
     for line in file:
          values = line.split()
          row = [float(value) for value in values]
          datos.append(row) 
data = np.array(datos)
#print(data)

#* Asigna los clusters a lugares random dentro de los limites de los puntos
cl = np.zeros((3,2))
for i in range(cantClusters): 
     cl[i,0]=np.random.uniform(min(data[:,0]), max(data[:,0]))
     cl[i,1]=np.random.uniform(min(data[:,1]), max(data[:,1]))
print("cl:")
print(cl)

#* Crea la matriz de asignacion de clusters inicializada en 0 (primer cl)
clData = np.zeros(data.shape[0])

#* Ciclo de asignacion de clusters
clNue = cl; 
primera = True
while np.array_equal(clNue,cl) == False or primera == True: 
    primera = False
    cl = clNue
    for i in range(data.shape[0]):
        clAct = clData[i]
        distAct = np.sqrt((data[i,0]-cl[clAct,0])**2 + (data[i,1]-cl[clAct,1])**2)
        for j in range(cantClusters): 
             if np.sqrt((data[i,0]-cl[j,0])**2 + (data[i,1]-cl[j,1])**2) < distAct: 
               distAct = np.sqrt((data[i,0]-cl[j,0])**2 + (data[i,1]-cl[j,1])**2) 
               clData[i] = j    
            

    for j in range(cantClusters): 
        nueX = 0
        nueY = 0
        cantPtos = 0
        for i in range(data.shape[0]): 
            if clData[i] == j: 
                nueX += data[i,0] 
                nueY += data[i,1] 
                cantPtos += 1 
        clNue[j,0] = nueX / cantPtos 
        clNue[j,1] = nueY / cantPtos 
print("cl nuevo:")
print(cl)
print("cldata: ")
print(clData)
    


plt.plot(data[:,0],data[:,1],'o', markersize=3)
plt.xlabel("coordenada x")
plt.ylabel("coordenada y")
plt.show()

