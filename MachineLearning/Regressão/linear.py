"""
A regressão linear simples é um método estatístico que modela a relação linear entre uma variável independente 
X e uma variável dependente Y.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dados de exemplo
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([2, 3, 4, 5, 6])

# Criando o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(X, Y)

# Coeficientes
intercepcao = modelo.intercept_
inclinação = modelo.coef_[0]

# Prevendo valores
Y_predito = modelo.predict(X)

# Visualizando a regressão linear
plt.scatter(X, Y, label='Dados Observados')
plt.plot(X, Y_predito, color='red', label='Regressão Linear')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
