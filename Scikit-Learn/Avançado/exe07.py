"""
Aprendizado Não Supervisionado com PCA (Análise de Componentes Principais):

Aplique PCA para redução de dimensionalidade em um conjunto de dados.
Visualize a variação explicada pelos componentes principais.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Carregando o conjunto de dados Iris
iris = load_iris()
X, y = iris.data, iris.target

# Padronizando as features para ter média zero e desvio padrão unitário (importante para o PCA)
X_standardized = StandardScaler().fit_transform(X)

# Aplicando PCA para redução de dimensionalidade
pca = PCA()
X_pca = pca.fit_transform(X_standardized)

# Visualizando a variação explicada pelos componentes principais
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Plotando a variação explicada acumulada
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
plt.xlabel('Número de Componentes Principais')
plt.ylabel('Variação Explicada Acumulada')
plt.title('Variação Explicada Acumulada pelos Componentes Principais')
plt.show()


"""
Utilizamos o conjunto de dados Iris.
Padronizamos as features para garantir média zero e desvio padrão unitário.
Aplicamos PCA para redução de dimensionalidade.
Visualizamos a variação explicada pelos componentes principais.


O gráfico mostrará como a variação explicada acumulada aumenta com o número de componentes principais. Isso pode ajudar a decidir quantos componentes principais reter para manter uma quantidade significativa de informação.
"""