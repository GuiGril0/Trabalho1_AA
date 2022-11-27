import numpy as np
import pandas as pd

class NaiveBayesUevora:
    alpha = float
    # construtor
    def __int__(self,alpha = 0.0 ):
        self.alpha = alpha
        self.colunas = list
        self.numPropriedades = {}      #por coluna
        self.probaXc = {}           #P(x|c)
        self.classes = {}           #P(c)
        self.probaX = {}            #P(x)

        self.xTreino = np.array
        self.yTreino = np.array
        self.treinoSize = int       #numero de linhas
        self.numColunas = int       #numero de colunas -1

    def fit(self, x, y):
        self.colunas = list(x.columns)
        self.xTreino = x
        self.yTreino = y
        self.treinoSize = x.shape[0]
        self.numColunas = x.shape[1]

        for atributo in self.colunas:
            self.probaXc[atributo] = {}
            self.probaX[atributo] = {}
            self.numPropriedades[atributo] = len(np.unique(self.xTreino[atributo]))

            for valorX in np.unique(self.xTreino[atributo]):
                self.probaX[atributo][valorX] = 0
                self.probaXc[atributo][valorX] = {}

                for valorY in np.unique(self.yTreino):
                    self.probaXc[atributo][valorX][valorY] = 0
                    self.classes[valorY] = 0

        # P(a)
        for valorY in np.unique(self.yTreino):
            nOcorrencias = sum(self.yTreino == valorY)
            #P(c) = ( nOcorrencias + alpha)/(total * (alpha * NPropriedades))
            self.classes[valorY] = (nOcorrencias + self.alpha) / \
               (self.treinoSize + (self.alpha * len(np.unique(self.yTreino))))

        # P(b|a)
        for atributo in self.colunas:

            for valorY in np.unique(self.yTreino):
                nOcorrY = sum(self.yTreino == valorY)
                nOcorrX_Y = self.xTreino[atributo][self.yTreino[\
                    self.yTreino == valorY].index].value_counts().to_dict()
                for valor, nOcorr in nOcorrX_Y.items():
                    self.probaXc[atributo][valor][valorY] = (nOcorr + self.alpha)/ \
                        (nOcorrY + (self.alpha * self.numPropriedades[atributo]))

        # P(b)
        for atributo in self.colunas:
            nOcorrX = self.xTreino[atributo].value_counts().to_dict()
            for valor, nOcorr in nOcorrX.items():
                self.probaX[atributo][valor] = (nOcorr + self.alpha)/ \
                   (self.treinoSize + self.alpha + self.numPropriedades[atributo])

    def predict(self, x):
        resultados = []
        x= np.array(x)
        for teste in x:
            possiveisResultados = {}
            for valorY in np.unique(self.yTreino):
                Pa = self.classes[valorY]
                Pba = 1
                Pb = 1
                for atributo, propriedade in zip(self.colunas, teste):
                    #print(f'{atributo},{propriedade},{valorY}')
                    Pba *= self.probaXc[atributo][propriedade][valorY]
                    Pb *= self.probaX[atributo][propriedade]
                possiveisResultados[valorY] = Pba * Pa  # (Pba*Pa)/Pb
            #print(possiveisResultados)
            resultado = max(possiveisResultados,key=lambda x: possiveisResultados[x])
            resultados.append(resultado)
       # print(resultados)
        return np.array(resultados)
    def acuracy_score(self, x, y):
        prev = self.predict(x)
        return round(float((sum(prev == y))/ float(len(y))*100),2)
    #def precision_score(self, x, y):




nb = NaiveBayesUevora()
file= pd.read_csv("Dados/breast-cancer-test2.csv")
x = file.drop([file.columns[-1]], axis= 1)
y = file[file.columns[-1]]
nb.__int__()
nb.fit(x,y)
#print(nb.precision_score(x, y))
print(nb.acuracy_score(x,y))