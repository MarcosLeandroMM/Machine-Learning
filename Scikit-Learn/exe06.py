"""
Exercício 6: Clustering com K-Means

Aplique o algoritmo K-Means a um conjunto de dados.
Experimente diferentes números de clusters e avalie a qualidade da clusterização.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# Criando um conjunto de dados fictício com 3 clusters
X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# Visualizando o conjunto de dados
plt.scatter(X[:, 0], X[:, 1], s=50, cmap='viridis')
plt.title("Conjunto de Dados Original")
plt.show()

# Aplicando o algoritmo K-Means com diferentes números de clusters
num_clusters = [2, 3, 4, 5, 6]
for n_clusters in num_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X)
    labels = kmeans.labels_
    
    # Visualizando os clusters
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    plt.title(f"K-Means com {n_clusters} Clusters")
    plt.show()

    # Avaliando a qualidade da clusterização com o índice de Silhouette
    silhouette_avg = silhouette_score(X, labels)
    print(f"Número de clusters: {n_clusters}, Índice de Silhouette: {silhouette_avg}")

"""
Criamos um conjunto de dados fictício com três clusters usando make_blobs.
Aplicamos o algoritmo K-Means com diferentes números de clusters.
Visualizamos os clusters resultantes.
Avaliamos a qualidade da clusterização usando o índice de Silhouette.

"""