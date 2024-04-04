"""
Exercício 7: Validação Cruzada

Realize a validação cruzada em um modelo de sua escolha.
Avalie o desempenho médio e a variância do modelo.

"""

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Carregar o conjunto de dados Iris
iris = load_iris()
X = iris.data

# Experimentar diferentes números de clusters
k_values = range(2, 8)
inertia_values = []

for k in k_values:
    # Criar o modelo K-Means
    kmeans = KMeans(n_clusters=k, random_state=42)
    
    # Treinar o modelo
    kmeans.fit(X)
    
    # Obter a inércia (soma das distâncias quadráticas dentro de cada cluster)
    inertia_values.append(kmeans.inertia_)

# Plotar a curva de cotovelo (Elbow Method)
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia')
plt.title('Curva de Cotovelo para K-Means')
plt.show()


'''

Neste código, estamos criando instâncias do modelo K-Means para diferentes valores de k (número de clusters), treinando o modelo e armazenando a inércia para cada k. Em seguida, plotamos a curva de cotovelo para ajudar a identificar o número ideal de clusters.

Ao observar a curva de cotovelo, você deve procurar o ponto onde a inércia começa a se estabilizar. Esse é frequentemente considerado o número ideal de clusters. No entanto, a escolha do número de clusters pode depender do contexto do seu problema específico.

Depois de determinar o número ideal de clusters, você pode treinar o modelo K-Means com esse valor e realizar análises mais detalhadas sobre a qualidade da clusterização, como visualizar os clusters ou avaliar métricas específicas de clusterização.

'''