"""
A regressão logística é usada para modelar a relação entre uma variável dependente binária (0 ou 1) e uma ou mais variáveis independentes. A função logística é utilizada para transformar uma combinação linear das variáveis independentes em uma probabilidade entre 0 e 1. 

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Dados de exemplo
X, Y = make_classification(n_samples=100, n_features=1, n_informative=1, n_redundant=0, random_state=42)

# Criando o modelo de regressão logística
modelo_logistico = LogisticRegression()
modelo_logistico.fit(X, Y)

# Prevendo probabilidades
probabilidades = modelo_logistico.predict_proba(X)[:, 1]

# Visualizando a regressão logística
plt.scatter(X, Y, label='Dados Observados')
plt.plot(X, probabilidades, color='red', label='Regressão Logística')
plt.xlabel('X')
plt.ylabel('Probabilidade(Y=1)')
plt.legend()
plt.show()
