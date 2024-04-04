"""

Regressão com Regressão Logística:

Utilize um conjunto de dados de regressão, como o conjunto de dados Boston Housing.
Ajuste um modelo de regressão logística e otimize os parâmetros.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Carregando o conjunto de dados Breast Cancer
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Dividindo o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizando as features (importante para a regressão logística)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Criando e treinando o modelo de regressão logística
logistic_regression_model = LogisticRegression(random_state=42)
logistic_regression_model.fit(X_train_scaled, y_train)

# Fazendo previsões
y_pred = logistic_regression_model.predict(X_test_scaled)

# Avaliando o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Exibindo resultados
print(f'Acurácia: {accuracy:.2f}')
print('Matriz de Confusão:')
print(conf_matrix)


"""
Utilizamos o conjunto de dados Breast Cancer.
Dividimos o conjunto de dados em conjuntos de treinamento e teste.
Padronizamos as features usando StandardScaler.
Criamos e treinamos um modelo de regressão logística.
Fazemos previsões e avaliamos o desempenho do modelo usando acurácia e matriz de confusão.
Lembre-se de que, para problemas de regressão, a regressão logística não é a escolha adequada. Para regressão, você pode considerar modelos como regressão linear, regressão de Ridge ou Lasso, entre outros.

"""
