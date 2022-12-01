import NaiveBayesUevora
import pandas as pd
import numpy as np

print("--------------------[Teste1]--------------------")
nb = NaiveBayesUevora.NaiveBayesUevora()
file = pd.read_csv("Dados/breast-cancer-train.csv")
x = file.drop([file.columns[-1]], axis= 1)
y = file[file.columns[-1]]
nb.__init__(1)
nb.fit(x,y)
file = pd.read_csv("Dados/breast-cancer-test.csv")
x1 = file.drop([file.columns[-1]], axis= 1)
y2 = file[file.columns[-1]]
print(f'Previsão: {nb.predict(x1)}')
print(f'Precisão: {nb.precision_score(x1, y2)}')
print(f'Exatidão: {nb.accuracy_score(x1,y2)}')

print("--------------------[Teste2]--------------------")
nb = NaiveBayesUevora.NaiveBayesUevora()
file = pd.read_csv("Dados/breast-cancer-train2.csv")
x = file.drop([file.columns[-1]], axis= 1)
y = file[file.columns[-1]]
nb.__init__(1)
nb.fit(x,y)
file = pd.read_csv("Dados/breast-cancer-test2.csv")
x1 = file.drop([file.columns[-1]], axis= 1)
y2 = file[file.columns[-1]]
print(f'Previsão: {nb.predict(x1)}')
print(f'Precisão: {nb.precision_score(x1, y2)}')
print(f'Exatidão: {nb.accuracy_score(x1,y2)}')

print("--------------------[Teste3]--------------------")
nb = NaiveBayesUevora.NaiveBayesUevora()
file = pd.read_csv("Dados/weather-nominal.csv")
x = file.drop([file.columns[-1]], axis= 1)
y = file[file.columns[-1]]
nb.__init__(1)
nb.fit(x,y)

x1 = [["rainy","hot","normal","t"]]
y2 =[["no"]]
print(f'Previsão: {nb.predict(x1)}')
