import numpy as np
import pandas as pd
import NaiveBayesUevora

from os import path

def main():
    while True:
        file = input("Introduza o nome do ficheiro: ")
        if path.isfile(file):
            break
        print("O input introduzido não corresponde a nenhum ficheiro existente!")

    data = get_content(file)

    while True:
        column = input("Escolha uma coluna: ")
        if check_column(data, column):
            break
        print("A coluna introduzida não corresponde a nenhuma coluna existente no ficheiro " + file + "!")

    while True:
        choice = input("Deseja atribuir um valor ao alpha? ")
        if choice.lower() in ["sim", "s", "não", "nao", "n"]:
            break
        print("Insira uma resposta que seja válida!")

    alpha = 0.0

    if choice.lower() in ["sim", "s"]:
        while True:
            try:
                alpha = float(input("Introduza um valor para o alpha: "))
                if alpha >= 0.0:
                    break
                raise ValueError
            except ValueError:
                print("Introduza um valor válido para o alpha (>= 0.0)!")

    X, y = process_data(data, column)

    nb = NaiveBayesUevora.NaiveBayesUevora()
    nb.__init__(alpha)
    nb.fit(X, y)

    while True:
        choice = input("Deseja utilizar terminal ou ficheiro? ")
        if choice.lower() in ["terminal", "t", "ficheiro", "ficheiros", "f"]:
            break
        print("Introduza uma resposta válida!")

    if choice.lower() in ["ficheiro", "ficheiros", "f"]:
        while True:
            test = input("Introduza o ficheiro de teste: ")
            if path.isfile(test):
                break
            print("O input introduzido não corresponde a um ficheiro existente!")
        d = get_content(test)
    else:
        d = []
        while True:
            print("Introduza as queries no terminal. Quando desejar terminar introduza 'exit' ou 'sair'.")
            aux = []
            for i in X:
                test = input(i + ": ")
                if test.lower() in ["exit", "sair"]:
                    break
                aux.append(test)
            if test.lower() in ["exit", "sair"]:
                break
            d.append(aux)

            i = 0;
            while i < len(data.loc[:, column]):
                line = X.iloc[i]
                if not all(item in line for item in d):
                    nb.addFit(d)
                ++i


    print(d)

    dX, dy = process_data(d, column)

    compare_datas(data, d)

    nb.predict(dX)


def get_content(filePath):
    return pd.read_csv(filePath)

def check_column(data, column):
    return column in data.columns

def process_data(data, c):
    #print(data)
    #X = data.drop([data.columns[c]], axis=1)
    X = data.drop(columns=[c], axis=1)
    y = data.loc[:, c]
    return X, y

def compare_datas(data, d):
     for i in range(data):
         print(i)

main()