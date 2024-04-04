"""
Redes Neurais com MLP (Perceptron de Múltiplas Camadas):

Aplique MLP para um conjunto de dados complexo, como o MNIST.
Ajuste a arquitetura da rede e otimize os hiperparâmetros.
Agrupamento Hierárquico

Para aplicar uma Rede Neural com MLP (Perceptron de Múltiplas Camadas) a um conjunto de dados complexo, como o MNIST (conjunto de dados de dígitos escritos à mão), você pode utilizar a biblioteca scikit-learn. No entanto, note que para tarefas mais complexas, muitas vezes é mais comum utilizar bibliotecas especializadas em deep learning, como TensorFlow ou PyTorch.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Carregando o conjunto de dados MNIST
mnist = fetch_openml('mnist_784')
X, y = mnist.data / 255.0, mnist.target.astype(int)

# Dividindo o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando e treinando o modelo MLP
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, random_state=42)
mlp_model.fit(X_train, y_train)

# Fazendo previsões
y_pred = mlp_model.predict(X_test)

# Avaliando o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Exibindo resultados
print(f'Acurácia: {accuracy:.2f}')
print('Matriz de Confusão:')
print(conf_matrix)


"""

Carregamos o conjunto de dados MNIST usando o scikit-learn.
Dividimos o conjunto de dados em conjuntos de treinamento e teste.
Criamos e treinamos um modelo MLP com uma camada oculta de 100 neurônios.
Fazemos previsões e avaliamos o desempenho do modelo usando acurácia e matriz de confusão.

"""