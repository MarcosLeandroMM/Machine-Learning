"""
Otimização de Hiperparâmetros com Grid Search:

Utilize Grid Search para encontrar os melhores hiperparâmetros para um modelo específico.

A otimização de hiperparâmetros usando Grid Search é uma técnica eficaz para encontrar a melhor combinação de hiperparâmetros para um modelo específico.

"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Carregando o conjunto de dados Iris
iris = load_iris()
X, y = iris.data, iris.target

# Dividindo o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definindo o classificador Random Forest
rf_classifier = RandomForestClassifier(random_state=42)

# Definindo os parâmetros para otimização usando Grid Search
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Realizando a busca em grade para otimização dos parâmetros
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Obtendo o melhor modelo após a busca em grade
best_rf_model = grid_search.best_estimator_

# Fazendo previsões no conjunto de teste
y_pred = best_rf_model.predict(X_test)

# Avaliando o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do Random Forest: {accuracy:.2f}')
print(f'Melhores Parâmetros: {grid_search.best_params_}')


"""
Utilizamos o conjunto de dados Iris.
Dividimos o conjunto de dados em conjuntos de treinamento e teste.
Definimos o classificador Random Forest.
Especificamos um conjunto de parâmetros para otimização usando a busca em grade (GridSearchCV).
Realizamos a busca em grade para otimização dos parâmetros.
Identificamos os melhores parâmetros e o modelo correspondente após a busca em grade.
Fazemos previsões no conjunto de teste e avaliamos o desempenho.


"""