"""
Detecção de Anomalias:

Aplique modelos de detecção de anomalias, como One-Class SVM, para identificar padrões anômalos em um conjunto de dados.

A detecção de anomalias é uma técnica usada para identificar padrões que são considerados raros ou diferentes do restante dos dados em um conjunto de dados. Um método comum para detecção de anomalias é o One-Class SVM (Support Vector Machine de uma classe), que é uma variação do SVM tradicional usada para encontrar uma fronteira que delimite a região onde a maioria dos dados está concentrada.



"""


from sklearn.svm import OneClassSVM
from sklearn.datasets import load_iris
import numpy as np

# Carregar conjunto de dados de exemplo (Iris Dataset)
X, _ = load_iris(return_X_y=True)

# Adicionar algumas anomalias artificiais ao conjunto de dados
anomalies = np.array([[6.5, 3.5, 8.0, 2.0], [7.5, 4.5, 5.0, 1.5]])
X_with_anomalies = np.vstack([X, anomalies])

# Treinar o modelo One-Class SVM
model = OneClassSVM(nu=0.05)  # Parâmetro nu controla a taxa de falsos positivos
model.fit(X_with_anomalies)

# Prever anomalias no conjunto de dados
predictions = model.predict(X_with_anomalies)

# Exibir as previsões (-1 indica anomalia, 1 indica instância normal)
print("Predictions:", predictions)
