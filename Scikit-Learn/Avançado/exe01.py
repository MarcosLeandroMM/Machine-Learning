"""

Classificação com SVM (Support Vector Machines):

Use o conjunto de dados Iris ou outro conjunto de dados adequado.
Experimente diferentes kernels (linear, polinomial, RBF) e otimize os parâmetros.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Carregando o conjunto de dados Iris
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Dividindo o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizando as features (importante para SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definindo os parâmetros do SVM
kernel_types = ['linear', 'poly', 'rbf']
C_values = [0.1, 1, 10]

# Iterando sobre diferentes kernels e parâmetros C
for kernel_type in kernel_types:
    for C_value in C_values:
        # Criando o classificador SVM
        svm_classifier = SVC(kernel=kernel_type, C=C_value, random_state=42)
        
        # Treinando o modelo
        svm_classifier.fit(X_train_scaled, y_train)
        
        # Fazendo previsões
        y_pred = svm_classifier.predict(X_test_scaled)
        
        # Avaliando a acurácia do modelo
        accuracy = accuracy_score(y_test, y_pred)
        
        # Exibindo resultados
        print(f"Kernel: {kernel_type}, C: {C_value}, Acurácia: {accuracy:.2f}")

"""

Carregamos o conjunto de dados Iris e dividimos em conjuntos de treinamento e teste.
Padronizamos as features usando StandardScaler.
Iteramos sobre diferentes kernels (linear, poly, rbf) e diferentes valores de C para otimizar os parâmetros do SVM.
Treinamos o modelo, fazemos previsões e avaliamos a acurácia para cada combinação de kernel e C.

"""