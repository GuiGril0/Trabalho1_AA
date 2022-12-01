import numpy as np
import pandas as pd

class NaiveBayesUevora:
    alpha = float

    #construtor

    def __init__(self,alpha = 0.0 ):
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
                    self.yTreino == valorY].index.values.tolist()].value_counts().to_dict()
                for valor, nOcorr in nOcorrX_Y.items():
                    self.probaXc[atributo][valor][valorY] = (nOcorr + self.alpha)/ \
                        (nOcorrY + (self.alpha * self.numPropriedades[atributo]))


    def addPropriety(self,atributo, propriedade):
        self.probaXc[atributo][propriedade] = {}
        for valorY in np.unique(self.yTreino):
            self.probaXc[atributo][propriedade][valorY] = 0
            nOcorrY = sum(self.yTreino == valorY)
            nOcorrX_Y = 0
            self.probaXc[atributo][propriedade][valorY] = (nOcorrX_Y + self.alpha) / \
                                                        (nOcorrY + (self.alpha * self.numPropriedades[atributo]+1))


    def predict(self, x):
        resultados = []
        x= np.array(x)
        for teste in x:
            possiveisResultados = {}
            for valorY in np.unique(self.yTreino):
                Pa = self.classes[valorY]
                Pba = 1
                for atributo, propriedade in zip(self.colunas, teste):
                    if propriedade not in np.unique(self.xTreino[atributo]):
                        self.addPropriety(atributo, propriedade)
                    Pba *= self.probaXc[atributo][propriedade][valorY]
                possiveisResultados[valorY] = Pba * Pa  # (Pba*Pa)/Pb
            resultado = max(possiveisResultados,key=lambda x: possiveisResultados[x])
            resultados.append(resultado)
        return np.array(resultados)


    def accuracy_score(self, x, y):
        prev = self.predict(x)
        return round(float((sum(prev == y))/ float(len(y))*100),2)


    def precision_score(self, x, y):
        resultados =[]
        prev = self.predict(x)
        for valorY in np.unique(self.yTreino):
            vp = 0
            fp = 0
            for vy , vyp in zip(y, prev):
                # print(f'{vy},{vyp},{valorY}')
                if vy == vyp and vyp == valorY:
                    vp +=1
                elif vyp != vy and vyp == valorY:
                    fp +=1
            if vp + fp == 0:
                resultados.append(0)
            else:
                resultados.append(vp / (vp + fp))
        return round(float(sum(resultados)/len(resultados)*100),2)


