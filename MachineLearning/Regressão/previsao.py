# Regressão Linear


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Dados de exemplo
X = np.random.rand(100, 1) * 10
Y = 2 * X + 1 + np.random.randn(100, 1) * 2

# Divisão dos dados
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.2, random_state=42)

# Criando e treinando o modelo de regressão linear
modelo_linear = LinearRegression()
modelo_linear.fit(X_treino, Y_treino)

# Prevendo valores
Y_predito = modelo_linear.predict(X_teste)

# Avaliação do modelo
mse = mean_squared_error(Y_teste, Y_predito)
print(f'MSE: {mse}')

# Visualização do modelo
plt.scatter(X_teste, Y_teste, label='Dados Observados')
plt.plot(X_teste, Y_predito, color='red', label='Regressão Linear')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
