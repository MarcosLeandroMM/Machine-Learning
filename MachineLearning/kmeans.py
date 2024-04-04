"""O algoritmo K-Means é um método de agrupamento (clustering) que particiona um conjunto de dados em
k clusters (grupos) com base nas características dos dados. Ele atribui cada ponto de dado ao cluster cujo centro (média) é mais próximo. Os passos principais do algoritmo são:
Inicialização dos Centroides: Escolher aleatoriamente k pontos de dados como centroides iniciais.
Atribuição aos Clusters: Atribuir cada ponto de dado ao centroide mais próximo, formando k clusters.
Atualização dos Centroides: Recalcular os centroides como as médias dos pontos atribuídos a cada cluster.
Repetição: Repetir os passos 2 e 3 até que não ocorram alterações significativas ou um número máximo de iterações seja atingido."""

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Dados de exemplo
X = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])

# Criando o modelo K-Means com 2 clusters
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# Obtendo os centroids e rótulos dos clusters
centroids = kmeans.cluster_centers_
rotulos = kmeans.labels_

# Visualização dos clusters e centroids
cores = ["g.", "r."]
for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], cores[rotulos[i]], markersize=10)
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
plt.show()

"""
Neste exemplo, o algoritmo K-Means agrupa os pontos de dados em dois clusters com base nas posições dos centroids. Os pontos são coloridos de acordo com seus rótulos de cluster, e os "x" representam os centroids.
"""