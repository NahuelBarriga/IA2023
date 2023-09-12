from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

datos= []
cantClusters=3

#* Pasa los datos del .txt a un array
with open("Clustering/samplesVDA1.txt") as file:
    for nbrLine, line in enumerate(file, 1):
        datos.append((nbrLine, float(line.strip()))) 
data = np.array(datos)
print(data)

#* Asigna los clusters a lugares random dentro de los limites de los puntos
cl = np.zeros((cantClusters,2))
for i in range(cantClusters): 
     cl[i,0]=np.random.uniform(min(data[:,0]), max(data[:,0]))
     cl[i,1]=np.random.uniform(min(data[:,1]), max(data[:,1]))

#* Crea la matriz de asignacion de clusters inicializada en 0 (primer cl)
clData = np.zeros(data.shape[0])

#* Ciclo de asignacion de clusters
clNue = cl.copy()
#print(cl)
primera = True
con=0



while True: 
    con += 1
    print(con) 
    primera = False
    cl = clNue.copy()
    for i in range(data.shape[0]):
        clAct = int(clData[i])
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
        if cantPtos != 0:
            clNue[j,0] = nueX / cantPtos 
            clNue[j,1] = nueY / cantPtos 
    print(clNue)
    if np.allclose(clNue,cl,atol =1e-6): 
        break

    # Borra la figura anterior
    plt.clf()

    # Define un conjunto de colores personalizados para cada grupo en 'clData'
    colormap = plt.cm.get_cmap('tab10', len(np.unique(clData)))
    # Crea una gráfica de dispersión asignando colores basados en 'clData'
    plt.scatter(data[:, 0], data[:, 1], c=clData, cmap=colormap, s=7)
    plt.scatter(cl[:, 0], cl[:, 1], c='black', marker='x')

    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    plt.title('Clustering')
    plt.pause(1)

plt.show()



    
#colors = ['red', 'green', 'blue', 'orange']
#plt.scatter(data[:,0],data[:,1], c=colors)
#plt.xlabel("coordenada x")
#plt.ylabel("coordenada y")
#plt.show()

