import numpy as np
import pandas as pd

from os import path

def main():
    file = input("Introduza o nome do ficheiro: ")
    while not path.isfile(file):
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
    nb.__init__(alpha)
    nb.fit(X,y)
    """
    while True:
        query = input("Insira a query: ")
        if query.lower() == "exit":
            break
        print(np.fromstring(query))
        nb.predict(np.array(query))
    exit(1)"""

def getContent(filePath):
    return pd.read_csv(filePath)

def check_column(data, column):
    print(data.columns.get_indexer(column))
    p
    return data.columns.get_indexer(column)

def processData(data, c):
    X = data.drop([data.columns[c]], axis=1)
    y = data[data.columns[c]]
    return X, y

class NaiveBayesUevora:
    alpha = 0

    #construtor
    def __init__(self,  alpha = 0):
        self.propriedades = list
        self.propNumVar = {}
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
        #print(self.propriedades)
        for prop in self.propriedades:
            self.numOcorrencias[prop] = {}
            self.PropriedadesProb[prop] = {}
            self.propNumVar[prop] = len(np.unique(self.x_treino[prop]))
            #print(f'{prop}--{len(np.unique(self.x_treino[prop]))}')

            for var  in np.unique(self.x_treino[prop]):
                self.PropriedadesProb[prop].update({var: 0})

                for var_final in np.unique(self.y_treino):
                    self.numOcorrencias[prop].update({var+"_"+var_final: 0})
                    self.clasProb.update({var_final: 0})

        for var_final in np.unique(self.y_treino):
            #print(var_final)
            count = sum(self.y_treino == var_final)
            self.clasProb[var_final] = (count + self.alpha)/(self.treinoSize + (self.alpha * len(np.unique(self.y_treino))  ) )
            #print((count + self.alpha)/(self.treinoSize + (self.alpha * len(np.unique(self.y_treino)))))
        for prop in self.propriedades:

            for var_final in np.unique(self.y_treino):
                count_total = sum(self.y_treino == var_final)
                prop_proba = self.x_treino[prop][self.y_treino[self.y_treino == var_final].index.values.tolist()].value_counts().to_dict()
            for prop_val, count in prop_proba.items():
                self.numOcorrencias[prop][prop_val+"_"+var_final] = (count +self.alpha) /(count_total + (self.alpha * self.propNumVar[prop]))

        for prop in self.propriedades:
            prop_vals = self.x_treino[prop].value_counts().to_dict()

            for prop_val, count in prop_vals.items():
                self.PropriedadesProb[prop][prop_val] = (count +self.alpha)/ (self.treinoSize + (self.alpha + self.propNumVar[prop] ))
    """
    função para previsões em função
    de um conjunto de dados de teste,
    com base no classificador definido
    na função fit(x, y)
    """
    def predict(self, X):
        results = []
        #X = np.array(X)
        print(type(X))
        for query in X:
            print(query)
            probs_outcome = {}
            for prop_vals in np.unique(self.y_treino):
                prior = self.clasProb[prop_vals]
                probabilidade = 1
                evidencia = 1

                for prop, prop_val in zip(self.propriedades, query):
                    probabilidade *= self.numOcorrencias[prop][prop_val+"_"+prop_vals]
                    evidencia *= self.PropriedadesProb[prop][prop_val]

                posterior = (probabilidade * prior) / evidencia

                probs_outcome[prop_vals] = posterior

            result = max(probs_outcome, key=lambda x: probs_outcome[x])
            results.append(result)
        return np.array(results)
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