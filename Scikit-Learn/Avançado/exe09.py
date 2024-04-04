"""
Avaliação de Desempenho com Curvas ROC e AUC:

Utilize métricas de avaliação de desempenho, como curvas ROC e AUC, para um modelo de classificação.


A Curva ROC (Receiver Operating Characteristic) e a AUC (Área sob a Curva) são métricas valiosas para avaliar o desempenho de modelos de classificação binária.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix

# Carregando o conjunto de dados Breast Cancer
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Dividindo o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando um modelo de Random Forest
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_scores = rf_classifier.predict_proba(X_test)[:, 1]

# Calculando a Curva ROC e a AUC
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Calculando a AUC diretamente usando a função roc_auc_score
roc_auc_score_value = roc_auc_score(y_test, y_scores)

# Plotando a Curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

# Exibindo a AUC diretamente
print(f'AUC: {roc_auc:.2f}')

"""
Utilizamos o conjunto de dados Breast Cancer.
Dividimos o conjunto de dados em conjuntos de treinamento e teste.
Criamos um modelo de Random Forest.
Fizemos previsões no conjunto de teste e calculamos a probabilidade dos rótulos positivos.
Calculamos a Curva ROC e a AUC.
Plotamos a Curva ROC e exibimos a AUC.
Exibimos a AUC diretamente usando a função roc_auc_score.



A Curva ROC mostra a relação entre a taxa de falsos positivos (FPR) e a taxa de verdadeiros positivos (TPR) para diferentes limiares de probabilidade. A AUC é uma métrica que resume a Curva ROC em um único valor, indicando a probabilidade de o modelo classificar uma instância positiva aleatória mais alta do que uma instância negativa aleatória.
"""