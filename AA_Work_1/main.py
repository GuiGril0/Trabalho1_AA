import numpy as np
import pandas as pd

import os.path
from os import path

def main():
    file = input("Introduza o nome do ficheiro (incluindo a extensão do mesmo): ")
    if not path.exists(file):
        print("O nome de ficheiro introduzido não existe!")
        main()
    data = getContent(file)

    #Escolher coluna aqui!!!!!!!!!!!!!!!!!!
    column = input("Escolha a coluna: ")

    print(data.columns['Play'])
    while check_column(data, column) == -1:
        column = input("Escolha a coluna: ")
    print(column)

    X, y = processData(data, column)

    nb = NaiveBayesUevora()
    nb.__init__()

def getContent(filePath):
    return pd.read_csv(filePath)

def check_column(data, column):
    print(data.columns.get_indexer(column))
    p
    return data.columns.get_indexer(column)

def processData(data, c):
    X = data.drop([data[c]], axis=1)
    y = data[data[c]]
    return X, y

class NaiveBayesUevora:
    alpha = 0

    #construtor
    def __init__(self,  alpha = 0):
        self.propriedades = list
        self.propNumVar = list
        self.numOcorrencias = {}
        self.clasProb= {}
        self.PropriedadesProb = {}

        self.x_treino = np.array
        self.y_treino = np.array
        self.treinoSize = int
        self.numPropriedades = int

    """
        função para gerar um classificador
        a partir de um conjunto de treino
        """
    def fit(self, x, y):
        self.propriedades = list(x.columns)
        self.x_treino = x
        self.y_treino = y
        self.treinoSize = x.shape[0] #numero de linhas
        self.numPropriedades = x.shape[1]   #numero de colunas

        for prop in self.propriedades:
            self.numPropriedades[prop] = {}
            self.PropriedadesProb[prop] = {}
            self.propNumVar[prop] = np.unique(self.x_treino[prop])

            for var  in np.unique(self.x_treino[prop]):
                self.PropriedadesProb[prop].update({var: 0})


    """
    função para previsões em função
    de um conjunto de dados de teste,
    com base no classificador definido
    na função fit(x, y)
    """
    #def predict(x):

    """
    função que calcula a exatidão
    de um classificador dado um
    conjunto de teste. Retorna um
    valor de tipo float
    """
    #def accuracy_score(x, y):

    """
    função que serve para calcular a
    precisão de um classificador dado
    um conjunto de teste. Retorna um
    valor de tipo float
    """
    #def precision_score(x, y):

main()