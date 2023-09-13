import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler
import time

cantCl=3
porcDatosTest = 0.20

def setData(porcDatosTest):
        datos= []
        datosTest = []      
        #* Pasa los datos del .txt a un array
        with open("TP1/samplesVDA4.txt") as file:
            for nbrLine, line in enumerate(file, 1):
                datos.append((nbrLine, float(line.strip()))) 
       #print(len(datos))

        for i in range(round(len(datos)*porcDatosTest)): 
            rand_idx = random.randrange(len(datos))
            datosTest.append(datos[rand_idx])
            datos.pop(rand_idx) 

        dataTest = np.array(datosTest)
        data = np.array(datos)
        #print(len(data))
        #print(len(dataTest))
        return data, dataTest



def kMeans(data, cantCl):
    #* Asigna los clusters a lugares random dentro de los limites de los puntos
    cl = np.zeros((cantCl,2))
    for i in range(cantCl): 
        cl[i,0]=np.random.uniform(min(data[:,0]), max(data[:,0]))
        cl[i,1]=np.random.uniform(min(data[:,1]), max(data[:,1]))
    #* Crea la matriz de asignacion de clusters inicializada en 0 (primer cl)
    clData = np.zeros(np.asanyarray(data).shape[0])
    #* Ciclo de asignacion de clusters
    clNue = cl.copy()
    largo = np.asanyarray(data).shape[0]
    while True:
        cl = clNue.copy()
        for i in range(largo):
            clAct = int(clData[i])
            distAct = np.sqrt((data[i,0]-cl[clAct,0])**2 + (data[i,1]-cl[clAct,1])**2)
            for j in range(cantCl): 
                if np.sqrt((data[i,0]-cl[j,0])**2 + (data[i,1]-cl[j,1])**2) < distAct: 
                    distAct = np.sqrt((data[i,0]-cl[j,0])**2 + (data[i,1]-cl[j,1])**2) 
                    clData[i] = j    
        for j in range(cantCl): 
            nueX = 0
            nueY = 0
            cantPtos = 0
            for i in range(largo): 
                if clData[i] == j: 
                    nueX += data[i,0] 
                    nueY += data[i,1] 
                    cantPtos += 1 
            if cantPtos != 0:
                clNue[j,0] = nueX / cantPtos 
                clNue[j,1] = nueY / cantPtos 
        #print(clNue)
        if np.allclose(clNue,cl,atol =1e-6): 
            break

    # Define un conjunto de colores personalizados para cada grupo en 'clData'
    colormap = plt.cm.get_cmap('tab10', len(np.unique(clData)))
    # Crea una gráfica de dispersión asignando colores basados en 'clData'
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=clData, cmap=colormap, s=7)
    plt.scatter(cl[:, 0], cl[:, 1], c='black', marker='x')
    plt.title("clusters")
    plt.xlabel('Tiempo')
    plt.ylabel('VDA')
    
    return clData, cl

def gaussmf(data, mean, sigma):
    return np.exp(-((data - mean)**2.) / (2 * sigma**2.))

class fisRule:
    def __init__(self, centroid, sigma):
        self.centroid = centroid
        self.sigma = sigma

class fisInput:
    def __init__(self, min,max, centroids):
        self.minValue = min
        self.maxValue = max
        self.centroids = centroids


    def view(self):
        x = np.linspace(self.minValue,self.maxValue,20)
        plt.figure()
        for m in self.centroids:
            s = (self.minValue-self.maxValue)/8**0.5
            y = gaussmf(x,m,s)
            plt.plot(x,y)

class fis:
    def __init__(self):
        self.rules=[]
        self.memberfunc = []
        self.inputs = []



    def genfis(self, data):

        start_time = time.time()
        #labels, cluster_center = kMeans(data, cantCl)
        clData, cl = kMeans(data, cantCl)
        #print("--- %s seconds ---" % (time.time() - start_time))
        n_clusters = len(cl)

        cl = cl[:,:-1]
        P = data[:,:-1]
        #T = data[:,-1]
        maxValue = np.max(P, axis=0)
        minValue = np.min(P, axis=0)

        self.inputs = [fisInput(maxValue[i], minValue[i],cl[:,i]) for i in range(len(maxValue))]
        self.rules = cl
        self.entrenar(data)

    def entrenar(self, data):
        P = data[:,:-1]
        T = data[:,-1]
        #___________________________________________
        # MINIMOS CUADRADOS (lineal)
        sigma = np.array([(i.maxValue-i.minValue)/np.sqrt(8) for i in self.inputs])
        f = [np.prod(gaussmf(P,cluster,sigma),axis=1) for cluster in self.rules]

        nivel_acti = np.array(f).T
        #print("nivel acti")
        #print(nivel_acti)
        sumMu = np.vstack(np.sum(nivel_acti,axis=1))
        #print("sumMu")
        #print(sumMu)
        P = np.c_[P, np.ones(len(P))]
        n_vars = P.shape[1]

        orden = np.tile(np.arange(0,n_vars), len(self.rules))
        acti = np.tile(nivel_acti,[1,n_vars])
        inp = P[:, orden]


        A = acti*inp/sumMu

        # A = np.zeros((N, 2*n_clusters))
        # for jdx in range(n_clusters):
        #     for kdx in range(nVar):
        #         A[:, jdx+kdx] = nivel_acti[:,jdx]*P[:,kdx]/sumMu
        #         A[:, jdx+kdx+1] = nivel_acti[:,jdx]/sumMu

        b = T

        solutions, residuals, rank, s = np.linalg.lstsq(A,b,rcond=None)
        self.solutions = solutions #.reshape(n_clusters,n_vars)
        #print(solutions)
        return 0

    def evalfis(self, data):
        sigma = np.array([(input.maxValue-input.minValue) for input in self.inputs])/np.sqrt(8)
        f = [np.prod(gaussmf(data,cluster,sigma),axis=1) for cluster in self.rules]
        nivel_acti = np.array(f).T
        sumMu = np.vstack(np.sum(nivel_acti,axis=1))

        P = np.c_[data, np.ones(len(data))]

        n_vars = P.shape[1]
        n_clusters = len(self.rules)

        orden = np.tile(np.arange(0,n_vars), n_clusters)
        acti = np.tile(nivel_acti,[1,n_vars])
        inp = P[:, orden]
        coef = self.solutions

        return np.sum(acti*inp*coef/sumMu,axis=1)


    def viewInputs(self):
        for input in self.inputs:
            input.view()

def calculoErr(r, dataTest, cantCl): #!esta mal
    MSE = 0
    for i in range(dataTest.shape[0]): 
        MSE += ((dataTest[i,1] - r[i])**2) 
        MSE = MSE / len(dataTest)
    print(MSE)
    return None

data, dataTest = setData(porcDatosTest) 

data_x = data[:,0]
data_y = data[:,1]

#plt.plot(data_x, data_y)
# plt.ylim(-20,20)
#plt.xlim(-7,7)

data = np.vstack((data_x, data_y)).T

fis2 = fis()
fis2.genfis(data)
fis2.viewInputs()
r = fis2.evalfis(np.vstack(data_x))

calculoErr(r, dataTest, cantCl)

#print(r)

plt.figure()
plt.plot(data_x,data_y)
plt.title('VDA x Tiempo')
plt.xlabel('Tiempo')
plt.ylabel('VDA')
plt.plot(data_x,r,linestyle='--')
plt.show()
fis2.solutions


#colors = ['red', 'green', 'blue', 'orange']
#plt.scatter(data[:,0],data[:,1], c=colors)
#plt.xlabel("coordenada x")
#plt.ylabel("coordenada y")
#plt.show()

