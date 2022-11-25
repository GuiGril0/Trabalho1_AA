import pandas as pd

def getContent(filePath):
    return pd.read_csv(filePath);

print(getContent('Dados/breast-cancer-test.csv'))