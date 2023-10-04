import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler
import time

porcDatosTest = 0
topeCL = 7

def setData(porcDatosTest):
        datos= []
        datosTest = []      
        #* Pasa los datos del .txt a un array
        with open("TP1/samplesVDA4.txt") as file:
            for nbrLine, line in enumerate(file, 1):
                datos.append((nbrLine*2.5, float(line.strip()))) 
       #print(len(datos))

        #for i in range(round(len(datos)*porcDatosTest)): 
        #    rand_idx = random.randrange(len(datos))
        #    datosTest.append(datos[rand_idx])
        #    datos.pop(rand_idx)     
        dataTest = np.array(datosTest)
        data = np.array(datos)
        #print(len(data))
        #print(len(dataTest))
        return data, dataTest


def normData(dataY):
    dataNorm = np.zeros((dataY.size))
    for i in range(dataY.size): 
        dataNorm[i] = (dataY[i] - min(dataY))/ (max(dataY) - min(dataY))
    return dataNorm


def kMeans(data, cantCl, grafica):
    #* Asigna los clusters a lugares random dentro de los limites de los puntos
    cl = np.zeros((cantCl,2))
    for i in range(cantCl): 
        rand_idx = random.randrange(len(data))
        cl[i,0]=data[rand_idx,0]
        cl[i,1]=data[rand_idx,1]
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
    if grafica:
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



    def genfis(self, data, cantCl, grafica):

        start_time = time.time()
        #labels, cluster_center = kMeans(data, cantCl)
        clData, cl = kMeans(data, cantCl, grafica)
        #print("--- %s seconds ---" % (time.time() - start_time))
        n_clusters = len(cl)

        cl = cl[:,-1:]
        P = data[:,-1:]
        #T = data[:,-1]
        maxValue = np.max(P, axis=0)
        minValue = np.min(P, axis=0)

        self.inputs = [fisInput(maxValue[i], minValue[i],cl[:,i]) for i in range(len(maxValue))]
        self.rules = cl
        self.entrenar(data)

    def entrenar(self, data):
        P = data[:,:-1]
        T = data[:,-1]
        Z = data[:,-1:]
        #___________________________________________
        # MINIMOS CUADRADOS (lineal)
        sigma = np.array([(i.maxValue-i.minValue)/np.sqrt(8) for i in self.inputs])
        f = [np.prod(gaussmf(Z,cluster,sigma),axis=1) for cluster in self.rules]

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

        solutions, residuals, rank, s = np.linalg.lstsq(A,b,rcond=None) #devuelve la solucion de los minimos cuadrados
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

def calculoErrTest(r, dataTest, cantCl): #!esta mal
    MSEtest = 0
    print(r)
    print(dataTest)
    for i in range(dataTest.shape[0]): 
        MSEtest += ((dataTest[i,1] - r[i])**2) 
    MSEtest = MSEtest / len(dataTest)
    print(MSEtest)
    plt.figure()
    plt.scatter(dataTest[:, 0], dataTest[:, 1], s=7)
    plt.title("dataTest")
    plt.xlabel('Tiempo')
    plt.ylabel('VDA')
    return None

def calculoErrTrain(r, data): 
    MSE = 0
    #print(r)
    #print(data) 
    for i in range(data.shape[0]): 
        MSE += ((data[i,1] - r[i])**2) 
    MSE = MSE / len(data)
    return MSE


def sobreMuest(data): 
    tope = data.shape[0]-1
    result = []
    for i in range(tope): 
        nuevo = np.array([(data[i,0] + data[i+1,0])/2, (data[i,1] + data[i+1,1])/2])
        result.append(data[i])
        result.append(nuevo)
    result.append(data[-1])
    result = np.vstack(result)
    indices_orden = np.argsort(result[:, 0])
    data_ordenada = result[indices_orden]
    return data_ordenada

def optCl(ClHist): 
    return np.argmin(ClHist)




data, dataTest = setData(porcDatosTest) 
data_x = data[:,0]
data_y = data[:,1]
data_y = normData(data_y)


#plt.plot(data_x, data_y)
#plt.show()
# plt.ylim(-20,20)
#plt.xlim(-7,7)

data = np.vstack((data_x, data_y)).T
MSEHist = []
ClHist = []


fis2 = fis()

grafica = False

for cantCl in range(3,topeCL):
    fis2.genfis(data,cantCl, grafica)
    if grafica:
        fis2.viewInputs()
    r = fis2.evalfis(np.vstack(data_y))
    MSEtrain = calculoErrTrain(r, data)
    print("CL: ", cantCl, "MSE: ",MSEtrain)
    MSEHist.append(MSEtrain)
    ClHist.append(cantCl) 

MSEHist = np.vstack(MSEHist)
ClHist = np.vstack(ClHist)
data = sobreMuest(data)

cantCl = optCl(MSEHist*ClHist) + 3
print("Cantidad de clusters optimo: ",cantCl)
grafica = True
fis2.genfis(data,cantCl, grafica)
fis2.viewInputs()
r = fis2.evalfis(np.vstack(data_y))
#MSEtrain = calculoErrTrain(r, data)

#minMSE = min(MSEHist)





plt.figure() 
plt.plot(ClHist,MSEHist) 
plt.title('cantCL x MSE')
plt.xlabel('cantCL')
plt.ylabel('MSE')


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

