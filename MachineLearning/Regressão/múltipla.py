"""
A regressão múltipla estende a ideia da regressão linear simples para incluir múltiplas variáveis independentes.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dados de exemplo
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
Y = np.array([2, 3, 4, 5, 6])

# Criando o modelo de regressão múltipla
modelo_multipla = LinearRegression()
modelo_multipla.fit(X, Y)

# Coeficientes
intercepcao = modelo_multipla.intercept_
coeficientes = modelo_multipla.coef_

# Prevendo valores
Y_predito = modelo_multipla.predict(X)

# Visualizando a regressão múltipla
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y, label='Dados Observados')
ax.scatter(X[:, 0], X[:, 1], Y_predito, color='red', label='Regressão Múltipla')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
plt.legend()
plt.show()
