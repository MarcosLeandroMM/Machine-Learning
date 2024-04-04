"""
alidação Cruzada:

Implemente a validação cruzada k-fold em um modelo de sua escolha.
Compare os resultados com a validação holdout.


A validação cruzada k-fold é uma técnica que divide o conjunto de dados em k partes, chamadas folds, e realiza o treinamento e a avaliação do modelo k vezes, cada vez utilizando uma parte diferente como conjunto de teste e as outras partes como conjunto de treinamento. Isso permite uma avaliação mais robusta do desempenho do modelo.
"""

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

# Carregando o conjunto de dados Breast Cancer
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Dividindo o conjunto de dados em treinamento e teste usando validação holdout
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando um modelo de Regressão Logística
logistic_model = LogisticRegression(random_state=42)

# Treinando o modelo usando validação holdout
logistic_model.fit(X_train, y_train)

# Avaliando o desempenho usando validação holdout
accuracy_holdout = logistic_model.score(X_test, y_test)
print(f'Acurácia (Holdout): {accuracy_holdout:.2f}')

# Avaliando o desempenho usando validação cruzada k-fold
k_fold_accuracy = cross_val_score(logistic_model, X, y, cv=5, scoring='accuracy')
print(f'Acurácia (Validação Cruzada k-fold): {np.mean(k_fold_accuracy):.2f} +/- {np.std(k_fold_accuracy):.2f}')


"""
Utilizamos o conjunto de dados Breast Cancer.
Dividimos o conjunto de dados em conjuntos de treinamento e teste usando a validação holdout.
Criamos um modelo de Regressão Logística.
Treinamos o modelo usando a validação holdout e avaliamos a acurácia.
Avaliamos o desempenho usando a validação cruzada k-fold e calculamos a média e o desvio padrão das acurácias.
A validação cruzada k-fold oferece uma avaliação mais robusta do desempenho do modelo, especialmente quando o conjunto de dados é limitado. Entretanto, ela pode ser computacionalmente mais intensiva. A escolha entre validação holdout e validação cruzada dependerá do tamanho do conjunto de dados, da disponibilidade computacional e dos objetivos específicos do projeto.








"""