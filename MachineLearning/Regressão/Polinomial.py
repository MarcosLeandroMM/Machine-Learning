"""
A regressão polinomial modela a relação entre uma variável independente e uma variável dependente utilizando uma equação polinomial de grau n.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Dados de exemplo
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([1, 4, 9, 16, 25])

# Transformação polinomial
grau_polinomio = 2
transformacao_polinomial = PolynomialFeatures(degree=grau_polinomio)
X_polinomial = transformacao_polinomial.fit_transform(X)

# Criando o modelo de regressão linear
modelo_polinomial = LinearRegression()
modelo_polinomial.fit(X_polinomial, Y)

# Prevendo valores
Y_predito = modelo_polinomial.predict(X_polinomial)

# Visualizando a regressão polinomial
plt.scatter(X, Y, label='Dados Observados')
plt.plot(X, Y_predito, color='red', label=f'Regressão Polinomial (Grau {grau_polinomio})')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
