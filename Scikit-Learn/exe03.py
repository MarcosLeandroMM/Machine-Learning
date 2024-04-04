"""
Exercício 3: Regressão Linear Simples

Crie um modelo de regressão linear simples usando o conjunto de dados de regressão fornecido pelo Scikit-learn (por exemplo, load_diabetes).
Treine o modelo e faça previsões.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes

# Carregar o conjunto de dados de diabetes
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar um modelo de regressão linear
model = LinearRegression()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o desempenho do modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

'''
Carregamos o conjunto de dados de diabetes usando load_diabetes do Scikit-learn.
Dividimos o conjunto de dados em conjuntos de treinamento e teste usando train_test_split.
Criamos um modelo de regressão linear simples usando LinearRegression.
Treinamos o modelo com os dados de treinamento usando o método fit.
Fazemos previsões no conjunto de teste usando o método predict.
Avaliamos o desempenho do modelo calculando o erro médio quadrático (MSE) usando mean_squared_error.

'''