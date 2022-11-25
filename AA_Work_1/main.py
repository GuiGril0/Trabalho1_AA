from sklearn.datasets import load_iris

import pandas as pd

class NaiveBayesUevora:
    alpha = 0

    #construtor
    def __init__(self, input, alpha = 0):
        self.data = set(input.split(","))
        self.alpha = alpha

    def choose_estimator(self):

    """
    função para gerar um classificador
    a partir de um conjunto de treino
    """
    def fit(x, y):

    """
    função para previsões em função
    de um conjunto de dados de teste,
    com base no classificador definido
    na função fit(x, y)
    """
    def predict(x):

    """
    função que calcula a exatidão
    de um classificador dado um
    conjunto de teste. Retorna um
    valor de tipo float
    """
    def accuracy_score(x, y):

    """
    função que serve para calcular a
    precisão de um classificador dado
    um conjunto de teste. Retorna um
    valor de tipo float
    """
    def precision_score(x, y):


def main:
