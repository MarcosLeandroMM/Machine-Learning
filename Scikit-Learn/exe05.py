"""
Exercício 5: Classificação com Random Forest

Crie um classificador de Random Forest para classificar um conjunto de dados de sua escolha.
Avalie o desempenho do modelo usando métricas como precisão, recall e matriz de confusão.
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar o conjunto de dados Iris
iris = load_iris()
X, y = iris.data, iris.target

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar um classificador de Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Treinar o modelo
rf_classifier.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = rf_classifier.predict(X_test)

# Avaliar o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Imprimir métricas de desempenho
print(f'Acurácia: {accuracy}')
print(f'Precisão: {precision}')
print(f'Recall: {recall}')
print('Matriz de Confusão:')
print(conf_matrix)

# Visualizar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()
